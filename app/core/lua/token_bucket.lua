-- Atomic token-bucket check-and-consume.
--
-- Runs as one Lua script (not GET-then-SET from Python) so concurrent
-- requests hitting different app replicas can't race on the same bucket —
-- Redis executes scripts single-threaded, so the read/refill/decrement/
-- write cycle below is effectively a single atomic operation.
--
-- KEYS[1] = bucket key
-- ARGV[1] = capacity  (max tokens / burst size)
-- ARGV[2] = refill_rate (tokens added per second)
-- ARGV[3] = requested(tokens this call wants to consume, usually 1)
-- ARGV[4] = ttl_seconds(expire the key once idle so buckets don't leak)
--
-- returns {allowed (0/1), tokens_remaining, retry_after_seconds}

local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local refill_rate = tonumber(ARGV[2])
local requested = tonumber(ARGV[3])
local ttl_seconds = tonumber(ARGV[4])

local data = redis.call("HMGET", key, "tokens", "ts")
local tokens = tonumber(data[1])
local last_ts = tonumber(data[2])

-- server-side clock (redis TIME), not the caller's, so skew between app
-- replicas can never distort the bucket's refill rate
local now = redis.call("TIME")
local now_s = tonumber(now[1]) + (tonumber(now[2]) / 1000000)

if tokens == nil then
    tokens = capacity
    last_ts = now_s
end

local elapsed = math.max(0, now_s - last_ts)
tokens = math.min(capacity, tokens + elapsed * refill_rate)

local allowed = 0
local retry_after = 0
if tokens >= requested then
    tokens = tokens - requested
    allowed = 1
else
    retry_after = (requested - tokens) / refill_rate
end

redis.call("HMSET", key, "tokens", tostring(tokens), "ts", tostring(now_s))
redis.call("EXPIRE", key, ttl_seconds)

return {allowed, tostring(tokens), tostring(retry_after)}
