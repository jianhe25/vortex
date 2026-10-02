-- Q0: One-hour range count. Measures zone pruning on the sorted time column.
SELECT COUNT(*) FROM hits
 WHERE "EventTime" >= TIMESTAMP '2013-07-15 12:00:00'
   AND "EventTime" < TIMESTAMP '2013-07-15 13:00:00';
-- Q1: One-day range with a tag aggregation.
SELECT "CounterID", COUNT(*) AS c FROM hits
 WHERE "EventTime" >= TIMESTAMP '2013-07-15 00:00:00'
   AND "EventTime" < TIMESTAMP '2013-07-16 00:00:00'
 GROUP BY "CounterID" ORDER BY c DESC LIMIT 10;
-- Q2: Newest rows. Measures reverse scan and TopK on the sort key.
SELECT "EventTime", "CounterID", "URL" FROM hits
 ORDER BY "EventTime" DESC LIMIT 10;
-- Q3: First rows after a point in time. Measures filter plus ordered limit.
SELECT "EventTime", "CounterID", "URL" FROM hits
 WHERE "EventTime" >= TIMESTAMP '2013-07-20 00:00:00'
 ORDER BY "EventTime" LIMIT 10;
-- Q4: Any rows inside a window. Measures filter plus limit without an ordering.
SELECT "WatchID", "EventTime", "URL" FROM hits
 WHERE "EventTime" >= TIMESTAMP '2013-07-20 00:00:00'
   AND "EventTime" < TIMESTAMP '2013-07-20 00:10:00'
 LIMIT 10;
-- Q5: Time bounds and row count. Answerable from file statistics alone.
SELECT MIN("EventTime"), MAX("EventTime"), COUNT(*) FROM hits;
-- Q6: Wide range count. Interior zones are fully covered and answerable from statistics.
SELECT COUNT(*) FROM hits
 WHERE "EventTime" >= TIMESTAMP '2013-07-10 00:00:00'
   AND "EventTime" < TIMESTAMP '2013-07-20 00:00:00';
-- Q7: Hourly buckets over one day.
SELECT DATE_TRUNC('hour', "EventTime") AS h, COUNT(*) AS c, COUNT(DISTINCT "UserID") AS u FROM hits
 WHERE "EventTime" >= TIMESTAMP '2013-07-15 00:00:00'
   AND "EventTime" < TIMESTAMP '2013-07-16 00:00:00'
 GROUP BY h ORDER BY h;
-- Q8: Daily buckets over the whole table.
SELECT DATE_TRUNC('day', "EventTime") AS d, COUNT(*) AS c, SUM("IsRefresh") AS refreshes FROM hits
 GROUP BY d ORDER BY d;
-- Q9: Minute buckets for one series over six hours.
SELECT DATE_TRUNC('minute', "EventTime") AS m, COUNT(*) AS c FROM hits
 WHERE "CounterID" = 62
   AND "EventTime" >= TIMESTAMP '2013-07-15 06:00:00'
   AND "EventTime" < TIMESTAMP '2013-07-15 12:00:00'
 GROUP BY m ORDER BY m;
-- Q10: Predicate on a function of the time column. Prunable only if the function has a falsifier.
SELECT COUNT(*) FROM hits
 WHERE DATE_TRUNC('hour', "EventTime") = TIMESTAMP '2013-07-15 12:00:00';
-- Q11: Hour-of-day profile across the whole table.
SELECT EXTRACT(HOUR FROM "EventTime") AS hr, COUNT(*) AS c FROM hits
 GROUP BY hr ORDER BY hr;
-- Q12: Timestamp plus interval between two columns.
SELECT COUNT(*) FROM hits
 WHERE "ClientEventTime" > "EventTime" + INTERVAL '1 hour';
-- Q13: Range relative to an anchor through interval arithmetic.
SELECT COUNT(*) FROM hits
 WHERE "EventTime" >= TIMESTAMP '2013-07-31 00:00:00' - INTERVAL '6 hours';
-- Q14: Latest point per series.
SELECT "CounterID", MAX("EventTime") AS last_seen FROM hits
 GROUP BY "CounterID" ORDER BY last_seen DESC LIMIT 10;
-- Q15: Newest rows for one series.
SELECT "EventTime", "URL" FROM hits
 WHERE "CounterID" = 62
 ORDER BY "EventTime" DESC LIMIT 10;
