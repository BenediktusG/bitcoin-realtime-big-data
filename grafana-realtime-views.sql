CREATE TABLE IF NOT EXISTS bigdata.bitcoin_orders_1h_agg (
    time DateTime,
    open_state AggregateFunction(argMin, Float64, DateTime),
    high_state AggregateFunction(max, Float64),
    low_state AggregateFunction(min, Float64),
    close_state AggregateFunction(argMax, Float64, DateTime),
    avg_close_state AggregateFunction(avg, Float64),
    volume_state AggregateFunction(sum, Float64)
) ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(time)
ORDER BY (time);

CREATE MATERIALIZED VIEW IF NOT EXISTS bigdata.bitcoin_orders_1h_mv
TO bigdata.bitcoin_orders_1h_agg AS
SELECT
    toStartOfInterval(time, INTERVAL 1 HOUR) AS time,
    argMinState(open, time) AS open_state,
    maxState(high) AS high_state,
    minState(low) AS low_state,
    argMaxState(close, time) AS close_state,
    avgState(close) AS avg_close_state,
    sumState(volume) AS volume_state
FROM bigdata.bitcoin_orders
GROUP BY time;

CREATE TABLE IF NOT EXISTS bigdata.bitcoin_predictions_1h_agg (
    time DateTime,
    close_prediction_state AggregateFunction(avg, Float64)
) ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(time)
ORDER BY (time);

CREATE MATERIALIZED VIEW IF NOT EXISTS bigdata.bitcoin_predictions_1h_mv
TO bigdata.bitcoin_predictions_1h_agg AS
SELECT
    toStartOfInterval(time, INTERVAL 1 HOUR) AS time,
    avgState(assumeNotNull(close_prediction)) AS close_prediction_state
FROM bigdata.bitcoin_predictions
WHERE close_prediction IS NOT NULL
GROUP BY time;

CREATE TABLE IF NOT EXISTS bigdata.bitcoin_features_1h_agg (
    time DateTime,
    close_delta_state AggregateFunction(avg, Float64),
    close_std_60_state AggregateFunction(avg, Float64),
    dist_to_mean_60_state AggregateFunction(avg, Float64),
    volume_sum_60_state AggregateFunction(avg, Float64)
) ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMM(time)
ORDER BY (time);

CREATE MATERIALIZED VIEW IF NOT EXISTS bigdata.bitcoin_features_1h_mv
TO bigdata.bitcoin_features_1h_agg AS
SELECT
    toStartOfInterval(time, INTERVAL 1 HOUR) AS time,
    avgState(close_delta) AS close_delta_state,
    avgState(close_std_60) AS close_std_60_state,
    avgState(dist_to_mean_60) AS dist_to_mean_60_state,
    avgState(volume_sum_60) AS volume_sum_60_state
FROM bigdata.bitcoin_features
GROUP BY time;

INSERT INTO bigdata.bitcoin_orders_1h_agg
SELECT
    toStartOfInterval(time, INTERVAL 1 HOUR) AS time,
    argMinState(open, time) AS open_state,
    maxState(high) AS high_state,
    minState(low) AS low_state,
    argMaxState(close, time) AS close_state,
    avgState(close) AS avg_close_state,
    sumState(volume) AS volume_state
FROM bigdata.bitcoin_orders
WHERE (SELECT count() FROM bigdata.bitcoin_orders_1h_agg) = 0
GROUP BY time;

INSERT INTO bigdata.bitcoin_predictions_1h_agg
SELECT
    toStartOfInterval(time, INTERVAL 1 HOUR) AS time,
    avgState(assumeNotNull(close_prediction)) AS close_prediction_state
FROM bigdata.bitcoin_predictions
WHERE close_prediction IS NOT NULL
  AND (SELECT count() FROM bigdata.bitcoin_predictions_1h_agg) = 0
GROUP BY time;

INSERT INTO bigdata.bitcoin_features_1h_agg
SELECT
    toStartOfInterval(time, INTERVAL 1 HOUR) AS time,
    avgState(close_delta) AS close_delta_state,
    avgState(close_std_60) AS close_std_60_state,
    avgState(dist_to_mean_60) AS dist_to_mean_60_state,
    avgState(volume_sum_60) AS volume_sum_60_state
FROM bigdata.bitcoin_features
WHERE (SELECT count() FROM bigdata.bitcoin_features_1h_agg) = 0
GROUP BY time;

CREATE TABLE IF NOT EXISTS bigdata.bitcoin_orders_1s_agg (
    time DateTime,
    close_state AggregateFunction(argMax, Float64, DateTime),
    volume_state AggregateFunction(sum, Float64)
) ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMMDD(time)
ORDER BY (time);

CREATE MATERIALIZED VIEW IF NOT EXISTS bigdata.bitcoin_orders_1s_mv
TO bigdata.bitcoin_orders_1s_agg AS
SELECT
    toStartOfInterval(time, INTERVAL 1 SECOND) AS time,
    argMaxState(close, time) AS close_state,
    sumState(volume) AS volume_state
FROM bigdata.bitcoin_orders
GROUP BY time;

CREATE TABLE IF NOT EXISTS bigdata.bitcoin_predictions_1s_agg (
    time DateTime,
    close_prediction_state AggregateFunction(avg, Float64)
) ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMMDD(time)
ORDER BY (time);

CREATE MATERIALIZED VIEW IF NOT EXISTS bigdata.bitcoin_predictions_1s_mv
TO bigdata.bitcoin_predictions_1s_agg AS
SELECT
    toStartOfInterval(time, INTERVAL 1 SECOND) AS time,
    avgState(assumeNotNull(close_prediction)) AS close_prediction_state
FROM bigdata.bitcoin_predictions
WHERE close_prediction IS NOT NULL
GROUP BY time;

CREATE TABLE IF NOT EXISTS bigdata.bitcoin_features_1s_agg (
    time DateTime,
    close_delta_state AggregateFunction(avg, Float64),
    close_std_60_state AggregateFunction(avg, Float64),
    dist_to_mean_60_state AggregateFunction(avg, Float64),
    volume_sum_60_state AggregateFunction(avg, Float64)
) ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMMDD(time)
ORDER BY (time);

CREATE MATERIALIZED VIEW IF NOT EXISTS bigdata.bitcoin_features_1s_mv
TO bigdata.bitcoin_features_1s_agg AS
SELECT
    toStartOfInterval(time, INTERVAL 1 SECOND) AS time,
    avgState(close_delta) AS close_delta_state,
    avgState(close_std_60) AS close_std_60_state,
    avgState(dist_to_mean_60) AS dist_to_mean_60_state,
    avgState(volume_sum_60) AS volume_sum_60_state
FROM bigdata.bitcoin_features
GROUP BY time;

INSERT INTO bigdata.bitcoin_orders_1s_agg
SELECT
    toStartOfInterval(time, INTERVAL 1 SECOND) AS time,
    argMaxState(close, time) AS close_state,
    sumState(volume) AS volume_state
FROM bigdata.bitcoin_orders
WHERE time >= now() - INTERVAL 24 HOUR
  AND (SELECT count() FROM bigdata.bitcoin_orders_1s_agg) = 0
GROUP BY time;

INSERT INTO bigdata.bitcoin_predictions_1s_agg
SELECT
    toStartOfInterval(time, INTERVAL 1 SECOND) AS time,
    avgState(assumeNotNull(close_prediction)) AS close_prediction_state
FROM bigdata.bitcoin_predictions
WHERE time >= now() - INTERVAL 24 HOUR
  AND close_prediction IS NOT NULL
  AND (SELECT count() FROM bigdata.bitcoin_predictions_1s_agg) = 0
GROUP BY time;

INSERT INTO bigdata.bitcoin_features_1s_agg
SELECT
    toStartOfInterval(time, INTERVAL 1 SECOND) AS time,
    avgState(close_delta) AS close_delta_state,
    avgState(close_std_60) AS close_std_60_state,
    avgState(dist_to_mean_60) AS dist_to_mean_60_state,
    avgState(volume_sum_60) AS volume_sum_60_state
FROM bigdata.bitcoin_features
WHERE time >= now() - INTERVAL 24 HOUR
  AND (SELECT count() FROM bigdata.bitcoin_features_1s_agg) = 0
GROUP BY time;

CREATE TABLE IF NOT EXISTS bigdata.bitcoin_orders_1s_7d_agg (
    time DateTime,
    close_state AggregateFunction(argMax, Float64, DateTime),
    volume_state AggregateFunction(sum, Float64)
) ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMMDD(time)
ORDER BY (time);

CREATE MATERIALIZED VIEW IF NOT EXISTS bigdata.bitcoin_orders_1s_7d_mv
TO bigdata.bitcoin_orders_1s_7d_agg AS
SELECT
    toStartOfInterval(time, INTERVAL 1 SECOND) AS time,
    argMaxState(close, time) AS close_state,
    sumState(volume) AS volume_state
FROM bigdata.bitcoin_orders
GROUP BY time;

CREATE TABLE IF NOT EXISTS bigdata.bitcoin_predictions_1s_7d_agg (
    time DateTime,
    close_prediction_state AggregateFunction(avg, Float64)
) ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMMDD(time)
ORDER BY (time);

CREATE MATERIALIZED VIEW IF NOT EXISTS bigdata.bitcoin_predictions_1s_7d_mv
TO bigdata.bitcoin_predictions_1s_7d_agg AS
SELECT
    toStartOfInterval(time, INTERVAL 1 SECOND) AS time,
    avgState(assumeNotNull(close_prediction)) AS close_prediction_state
FROM bigdata.bitcoin_predictions
WHERE close_prediction IS NOT NULL
GROUP BY time;

CREATE TABLE IF NOT EXISTS bigdata.bitcoin_features_1s_7d_agg (
    time DateTime,
    close_delta_state AggregateFunction(avg, Float64),
    close_std_60_state AggregateFunction(avg, Float64),
    dist_to_mean_60_state AggregateFunction(avg, Float64),
    volume_sum_60_state AggregateFunction(avg, Float64)
) ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMMDD(time)
ORDER BY (time);

CREATE MATERIALIZED VIEW IF NOT EXISTS bigdata.bitcoin_features_1s_7d_mv
TO bigdata.bitcoin_features_1s_7d_agg AS
SELECT
    toStartOfInterval(time, INTERVAL 1 SECOND) AS time,
    avgState(close_delta) AS close_delta_state,
    avgState(close_std_60) AS close_std_60_state,
    avgState(dist_to_mean_60) AS dist_to_mean_60_state,
    avgState(volume_sum_60) AS volume_sum_60_state
FROM bigdata.bitcoin_features
GROUP BY time;

INSERT INTO bigdata.bitcoin_orders_1s_7d_agg
SELECT
    toStartOfInterval(time, INTERVAL 1 SECOND) AS time,
    argMaxState(close, time) AS close_state,
    sumState(volume) AS volume_state
FROM bigdata.bitcoin_orders
WHERE time >= now() - INTERVAL 7 DAY
  AND (SELECT count() FROM bigdata.bitcoin_orders_1s_7d_agg) = 0
GROUP BY time;

INSERT INTO bigdata.bitcoin_predictions_1s_7d_agg
SELECT
    toStartOfInterval(time, INTERVAL 1 SECOND) AS time,
    avgState(assumeNotNull(close_prediction)) AS close_prediction_state
FROM bigdata.bitcoin_predictions
WHERE time >= now() - INTERVAL 7 DAY
  AND close_prediction IS NOT NULL
  AND (SELECT count() FROM bigdata.bitcoin_predictions_1s_7d_agg) = 0
GROUP BY time;

INSERT INTO bigdata.bitcoin_features_1s_7d_agg
SELECT
    toStartOfInterval(time, INTERVAL 1 SECOND) AS time,
    avgState(close_delta) AS close_delta_state,
    avgState(close_std_60) AS close_std_60_state,
    avgState(dist_to_mean_60) AS dist_to_mean_60_state,
    avgState(volume_sum_60) AS volume_sum_60_state
FROM bigdata.bitcoin_features
WHERE time >= now() - INTERVAL 7 DAY
  AND (SELECT count() FROM bigdata.bitcoin_features_1s_7d_agg) = 0
GROUP BY time;

CREATE TABLE IF NOT EXISTS bigdata.bitcoin_realtime_predictions_1s_7d_agg (
    time DateTime,
    close_prediction_state AggregateFunction(avg, Float64)
) ENGINE = AggregatingMergeTree()
PARTITION BY toYYYYMMDD(time)
ORDER BY (time);

CREATE MATERIALIZED VIEW IF NOT EXISTS bigdata.bitcoin_realtime_predictions_1s_7d_mv
TO bigdata.bitcoin_realtime_predictions_1s_7d_agg AS
SELECT
    toStartOfInterval(time, INTERVAL 1 SECOND) AS time,
    avgState(close_prediction) AS close_prediction_state
FROM bigdata.bitcoin_realtime_predictions
GROUP BY time;

INSERT INTO bigdata.bitcoin_realtime_predictions_1s_7d_agg
SELECT
    toStartOfInterval(time, INTERVAL 1 SECOND) AS time,
    avgState(close_prediction) AS close_prediction_state
FROM bigdata.bitcoin_realtime_predictions
WHERE time >= now() - INTERVAL 7 DAY
  AND (SELECT count() FROM bigdata.bitcoin_realtime_predictions_1s_7d_agg) = 0
GROUP BY time;
