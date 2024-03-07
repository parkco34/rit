#!/usr/bin/env python
import pandas as pd
from sqlalchemy import text, create_engine

QUERY = """
  SELECT geoNetwork.city AS City,
  SUM(CASE WHEN geoNetwork.city IN ('Seattle', 'Austin') THEN totals.pageviews ELSE 0 END) AS TotalPageviews,
  SUM(CASE WHEN geoNetwork.city IN ('Seattle', 'Austin') THEN totals.transactions ELSE 0 END) AS TotalTransactions,
  AVG(CASE WHEN geoNetwork.city IN ('Seattle', 'Austin') THEN totals.timeOnSite ELSE NULL END) AS AvgTimeOnSite,
  SUM(CASE WHEN geoNetwork.city IN ('Seattle', 'Austin') THEN totals.newVisits ELSE 0 END) AS NewVisits,
  AVG(CASE WHEN geoNetwork.city IN ('Seattle', 'Austin') THEN totals.sessionQualityDim ELSE NULL END) AS AvgSessionQuality,
  SUM(CASE WHEN geoNetwork.city IN ('Seattle', 'Austin') THEN totals.totalTransactionRevenue ELSE 0 END) AS TotalTransactionRevenue,
  SUM(CASE WHEN geoNetwork.city IN ('Seattle', 'Austin') THEN totals.transactionRevenue ELSE 0 END) / COUNT(CASE WHEN geoNetwork.city IN ('Seattle', 'Austin') AND totals.transactions IS NOT NULL THEN totals.transactions ELSE NULL END) AS AverageOrderValue,
  (SUM(CASE WHEN geoNetwork.city IN ('Seattle', 'Austin') THEN totals.transactions ELSE 0 END) / SUM(CASE WHEN geoNetwork.city IN ('Seattle', 'Austin') THEN totals.visits ELSE 0 END)) * 100 AS ConversionRate,
  SUM(CASE WHEN geoNetwork.city IN ('Seattle', 'Austin') THEN totals.transactionRevenue ELSE 0 END) / COUNT(DISTINCT CASE WHEN geoNetwork.city IN ('Seattle', 'Austin') THEN fullVisitorId ELSE NULL END) AS ARPU
FROM
  `bigquery-public-data.google_analytics_sample.ga_sessions_*`
WHERE
  _TABLE_SUFFIX BETWEEN '20160801' AND '20170801'
  AND geoNetwork.city IN ('Seattle', 'Austin')
GROUP BY
  City
ORDER BY
  City;

"""


db = create_engine(
    "bigquery://",
    credentials_path = "thing"
)
#
df = pd.read_sql(QUERY, con=db)
breakpoint()