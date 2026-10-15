-- Q1 — WHERE then HAVING
SELECT customer_id, SUM(order_amount) AS total_large_order_spend
FROM orders
WHERE status = 'completed' AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000
ORDER BY customer_id;

-- Q2 — Three-table revenue join
SELECT c.city, p.category, SUM(o.order_amount) AS revenue_inr
FROM orders AS o
JOIN customers AS c ON c.customer_id = o.customer_id
JOIN products AS p ON p.product_id = o.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY c.city, p.category;

-- Q3 — Products above average catalogue price
SELECT product_id, product_name, category, price_inr
FROM products
WHERE price_inr > (SELECT AVG(price_inr) FROM products)
ORDER BY price_inr DESC, product_id;

-- Q4 — Customers without a completed order
SELECT c.customer_id, c.city
FROM customers AS c
LEFT JOIN orders AS o
  ON o.customer_id = c.customer_id AND o.status = 'completed'
WHERE o.order_id IS NULL
ORDER BY c.customer_id;

-- Q5 — Customer spend above the average customer spend
WITH customer_spend AS (
  SELECT c.customer_id, c.city, SUM(o.order_amount) AS total_spend
  FROM customers AS c
  JOIN orders AS o ON o.customer_id = c.customer_id AND o.status = 'completed'
  GROUP BY c.customer_id, c.city
),
average_spend AS (
  SELECT AVG(total_spend) AS average_total FROM customer_spend
)
SELECT customer_spend.customer_id, customer_spend.city, customer_spend.total_spend
FROM customer_spend CROSS JOIN average_spend
WHERE customer_spend.total_spend > average_spend.average_total
ORDER BY customer_spend.customer_id;

-- Q6 — City ranking window functions
WITH customer_spend AS (
  SELECT c.customer_id, c.city, SUM(o.order_amount) AS total_spend
  FROM customers AS c
  JOIN orders AS o ON o.customer_id = c.customer_id
  WHERE o.status = 'completed'
  GROUP BY c.customer_id, c.city
)
SELECT customer_id, city, total_spend,
       RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS spend_rank,
       DENSE_RANK() OVER (PARTITION BY city ORDER BY total_spend DESC) AS dense_spend_rank,
       ROW_NUMBER() OVER (PARTITION BY city ORDER BY total_spend DESC, customer_id) AS city_row_number
FROM customer_spend
ORDER BY city, spend_rank, customer_id;

-- Q7 — Daily completed revenue and cumulative revenue
WITH daily_revenue AS (
  SELECT date(order_ts) AS day, SUM(order_amount) AS revenue
  FROM orders
  WHERE status = 'completed'
  GROUP BY date(order_ts)
)
SELECT day, revenue,
       SUM(revenue) OVER (ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
         AS cumulative_revenue
FROM daily_revenue
ORDER BY day;
