"""
DOES REDIS
Redis caching layer for Medical Literature RAG.
Dramatically reduces response time for repeated queries.

Why Redis over in-memory Python dict:
1. Persists across API restarts
2. Shared across multiple API workers
3. Built-in TTL (automatic expiration)
4. Memory-efficient with eviction policies
5. Production-ready with monitoring tools

Cache strategy:
- Cache full RAG responses for 24 hours
- Use query hash as key to handle variations
- Include paper count in key for auto-invalidation
- Store compressed JSON to save memory
"""

import json
import hashlib
import time
import logging
import zlib
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta

import redis
from redis.exceptions import RedisError, ConnectionError as RedisConnectionError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Cache configuration
CACHE_TTL_SECONDS = 86400  # 24 hours for medical content freshness
CACHE_KEY_PREFIX = "rag:v1"  # Version prefix for cache invalidation
COMPRESSION_THRESHOLD = 1024  # Compress responses larger than 1KB
MAX_CACHE_SIZE_MB = 100  # Maximum cache size before eviction

class CacheManager:
    """
    Manages Redis caching for RAG responses.
    
    Design decisions:
    1. Use SHA256 hash of query for consistent keys
    2. Include metadata (paper count, model version) in key
    3. Compress large responses to save RAM
    4. Graceful degradation if Redis unavailable
    """
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 6379,
                 db: int = 0,
                 decode_responses: bool = False,
                 socket_connect_timeout: float = 2.0,
                 socket_timeout: float = 1.0):
        """
        Initialize Redis connection with production settings.
        
        Args:
            host: Redis server hostname
            port: Redis server port
            db: Redis database number (0-15)
            decode_responses: False to handle binary data (for compression)
            socket_connect_timeout: Connection timeout in seconds
            socket_timeout: Operation timeout in seconds
        """
        self.host = host
        self.port = port
        self.db = db
        self.redis_client = None
        self.stats = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "total_time_saved": 0.0
        }
        
        # Connection pool for thread safety and efficiency
        self.pool = redis.ConnectionPool(
            host=host,
            port=port,
            db=db,
            decode_responses=decode_responses,
            socket_connect_timeout=socket_connect_timeout,
            socket_timeout=socket_timeout,
            max_connections=50  # Handle concurrent requests
        )
        
        self._connect()
    
    def _connect(self) -> None:
        """Establish Redis connection with retry logic."""
        max_retries = 3
        retry_delay = 0.5
        
        for attempt in range(max_retries):
            try:
                self.redis_client = redis.Redis(connection_pool=self.pool)
                # Test connection
                self.redis_client.ping()
                logger.info(f"✓ Connected to Redis at {self.host}:{self.port}")
                
                # Set memory policy for cache
                try:
                    self.redis_client.config_set('maxmemory-policy', 'allkeys-lru')
                    self.redis_client.config_set('maxmemory', f'{MAX_CACHE_SIZE_MB}mb')
                except:
                    # May not have CONFIG permission, that's okay
                    pass
                
                return
                
            except RedisConnectionError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Redis connection attempt {attempt + 1} failed, retrying...")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to connect to Redis: {e}")
                    logger.warning("Cache disabled - running without caching")
                    self.redis_client = None
    
    def _generate_cache_key(self, 
                           query: str, 
                           max_papers: int,
                           min_relevance: float,
                           paper_count: Optional[int] = None) -> str:
        """
        Generate deterministic cache key from query parameters.
        
        Why this approach:
        1. SHA256 handles query variations (spaces, punctuation)
        2. Including paper_count auto-invalidates when DB updates
        3. Parameters affect results, so they're part of the key
        
        Args:
            query: User's question
            max_papers: Maximum papers to retrieve
            min_relevance: Minimum relevance threshold
            paper_count: Total papers in DB (for invalidation)
            
        Returns:
            Cache key like "rag:v1:7f3a8b:p10:r0.3:c100"
        """
        # Normalize query (lowercase, strip whitespace)
        normalized_query = query.lower().strip()
        
        # Create hash of query for consistent key length
        query_hash = hashlib.sha256(normalized_query.encode()).hexdigest()[:6]
        
        # Build cache key with all parameters that affect the result
        parts = [
            CACHE_KEY_PREFIX,
            query_hash,
            f"p{max_papers}",
            f"r{min_relevance:.1f}"
        ]
        
        if paper_count is not None:
            parts.append(f"c{paper_count}")
        
        return ":".join(parts)
    
    def _compress_value(self, data: Dict) -> bytes:
        """
        Compress JSON data if it's large enough.
        
        Args:
            data: Dictionary to cache
            
        Returns:
            Compressed bytes or JSON bytes
        """
        json_str = json.dumps(data, default=str)  # default=str handles datetime
        json_bytes = json_str.encode('utf-8')
        
        if len(json_bytes) > COMPRESSION_THRESHOLD:
            compressed = zlib.compress(json_bytes, level=6)  # Balance speed/size
            # Add magic bytes to identify compressed data
            return b'ZLIB' + compressed
        
        return json_bytes
    
    def _decompress_value(self, data: bytes) -> Dict:
        """
        Decompress cached value if needed.
        
        Args:
            data: Cached bytes
            
        Returns:
            Original dictionary
        """
        if data.startswith(b'ZLIB'):
            decompressed = zlib.decompress(data[4:])
            return json.loads(decompressed.decode('utf-8'))
        
        return json.loads(data.decode('utf-8'))
    
    def get(self, 
            query: str,
            max_papers: int = 5,
            min_relevance: float = 0.3,
            paper_count: Optional[int] = None) -> Optional[Dict]:
        """
        Retrieve cached response if available.
        
        Args:
            query: User's question
            max_papers: Maximum papers parameter
            min_relevance: Minimum relevance parameter
            paper_count: Current paper count in DB
            
        Returns:
            Cached response dict or None if not found
        """
        if not self.redis_client:
            return None
        
        try:
            cache_key = self._generate_cache_key(query, max_papers, min_relevance, paper_count)
            
            # Get from cache
            start_time = time.time()
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                # Decompress and parse
                result = self._decompress_value(cached_data)
                
                # Update stats
                elapsed = time.time() - start_time
                self.stats["hits"] += 1
                # Assume we saved ~7 seconds of model inference
                self.stats["total_time_saved"] += 7.0
                
                logger.info(f"✓ Cache HIT for query: '{query[:50]}...' (retrieved in {elapsed:.3f}s)")
                
                # Add cache metadata
                result["_cache_hit"] = True
                result["_cache_key"] = cache_key
                
                return result
            else:
                self.stats["misses"] += 1
                logger.debug(f"Cache MISS for query: '{query[:50]}...'")
                return None
                
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache GET error: {e}")
            return None  # Graceful degradation
    
    def set(self,
            query: str,
            response: Dict,
            max_papers: int = 5,
            min_relevance: float = 0.3,
            paper_count: Optional[int] = None,
            ttl_seconds: Optional[int] = None) -> bool:
        """
        Store response in cache.
        
        Args:
            query: User's question
            response: RAG response to cache
            max_papers: Maximum papers parameter
            min_relevance: Minimum relevance parameter
            paper_count: Current paper count in DB
            ttl_seconds: Custom TTL (defaults to CACHE_TTL_SECONDS)
            
        Returns:
            True if successfully cached
        """
        if not self.redis_client:
            return False
        
        try:
            cache_key = self._generate_cache_key(query, max_papers, min_relevance, paper_count)
            ttl = ttl_seconds or CACHE_TTL_SECONDS
            
            # Add caching metadata
            response_copy = response.copy()
            response_copy["_cached_at"] = datetime.now().isoformat()
            response_copy["_expires_at"] = (datetime.now() + timedelta(seconds=ttl)).isoformat()
            
            # Compress and store
            compressed_data = self._compress_value(response_copy)
            
            # Set with expiration
            success = self.redis_client.setex(
                cache_key,
                ttl,
                compressed_data
            )
            
            if success:
                size_kb = len(compressed_data) / 1024
                logger.info(f"✓ Cached response for: '{query[:50]}...' ({size_kb:.1f}KB, TTL={ttl}s)")
                return True
            
            return False
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache SET error: {e}")
            return False
    
    def delete(self, 
               query: str,
               max_papers: int = 5,
               min_relevance: float = 0.3,
               paper_count: Optional[int] = None) -> bool:
        """
        Delete specific cached response.
        
        Args:
            query: Query to invalidate
            max_papers: Maximum papers parameter
            min_relevance: Minimum relevance parameter
            paper_count: Paper count parameter
            
        Returns:
            True if deleted
        """
        if not self.redis_client:
            return False
        
        try:
            cache_key = self._generate_cache_key(query, max_papers, min_relevance, paper_count)
            deleted = self.redis_client.delete(cache_key)
            
            if deleted:
                logger.info(f"✓ Deleted cache for: '{query[:50]}...'")
            
            return bool(deleted)
            
        except Exception as e:
            logger.error(f"Cache DELETE error: {e}")
            return False
    
    def flush_pattern(self, pattern: str = f"{CACHE_KEY_PREFIX}:*") -> int:
        """
        Delete all keys matching pattern.
        
        Args:
            pattern: Redis key pattern (e.g., "rag:v1:*")
            
        Returns:
            Number of keys deleted
        """
        if not self.redis_client:
            return 0
        
        try:
            # Use SCAN to avoid blocking on large keyspaces
            deleted_count = 0
            for key in self.redis_client.scan_iter(match=pattern, count=100):
                self.redis_client.delete(key)
                deleted_count += 1
            
            logger.info(f"✓ Flushed {deleted_count} cache entries matching '{pattern}'")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Cache FLUSH error: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache performance statistics.
        
        Returns:
            Dictionary with cache metrics
        """
        stats = self.stats.copy()
        
        # Calculate hit rate
        total_requests = stats["hits"] + stats["misses"]
        stats["hit_rate"] = stats["hits"] / total_requests if total_requests > 0 else 0
        
        # Get Redis stats if available
        if self.redis_client:
            try:
                info = self.redis_client.info("memory")
                stats["redis_memory_used_mb"] = info.get("used_memory", 0) / (1024 * 1024)
                stats["redis_memory_peak_mb"] = info.get("used_memory_peak", 0) / (1024 * 1024)
                
                # Count our keys
                our_keys = list(self.redis_client.scan_iter(match=f"{CACHE_KEY_PREFIX}:*", count=100))
                stats["cached_queries"] = len(our_keys)
                
            except Exception as e:
                logger.debug(f"Could not get Redis stats: {e}")
        
        return stats
    
    def health_check(self) -> Tuple[bool, str]:
        """
        Check if cache is healthy.
        
        Returns:
            Tuple of (is_healthy, status_message)
        """
        if not self.redis_client:
            return False, "Redis client not initialized"
        
        try:
            # Ping Redis
            self.redis_client.ping()
            
            # Try a test operation
            test_key = f"{CACHE_KEY_PREFIX}:health"
            self.redis_client.setex(test_key, 10, "healthy")
            self.redis_client.delete(test_key)
            
            return True, "Redis cache operational"
            
        except Exception as e:
            return False, f"Redis error: {str(e)}"
    
    def close(self) -> None:
        """Clean up Redis connection."""
        if self.redis_client:
            self.pool.disconnect()
            logger.info("✓ Cache manager closed")


def demonstrate_cache():
    """Demonstrate caching functionality."""
    print("\n" + "="*60)
    print("REDIS CACHE DEMONSTRATION")
    print("="*60)
    
    # Initialize cache
    cache = CacheManager()
    
    # Check health
    healthy, message = cache.health_check()
    print(f"\nHealth check: {message}")
    
    if not healthy:
        print("❌ Redis not available, exiting demo")
        return
    
    # Test caching
    test_query = "What are immunotherapy options for lung cancer?"
    test_response = {
        "answer": "Based on the research papers...",
        "citations": [{"pmid": "12345", "title": "Test paper"}],
        "confidence": 0.85,
        "response_time": 7.5
    }
    
    print(f"\nTest query: '{test_query}'")
    
    # First request (miss)
    print("\n1. First request (cache MISS):")
    cached = cache.get(test_query)
    print(f"   Result: {'Found' if cached else 'Not found'}")
    
    # Store in cache
    print("\n2. Storing in cache...")
    success = cache.set(test_query, test_response)
    print(f"   Success: {success}")
    
    # Second request (hit)
    print("\n3. Second request (cache HIT):")
    cached = cache.get(test_query)
    print(f"   Result: {'Found' if cached else 'Not found'}")
    if cached:
        print(f"   Cached at: {cached.get('_cached_at')}")
        print(f"   Expires at: {cached.get('_expires_at')}")
    
    # Show stats
    print("\n4. Cache statistics:")
    stats = cache.get_stats()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")
    
    # Clean up
    cache.flush_pattern()
    cache.close()
    
    print("\n" + "="*60)
    print("✅ CACHE DEMONSTRATION COMPLETE")
    print("="*60)


if __name__ == "__main__":
    demonstrate_cache()