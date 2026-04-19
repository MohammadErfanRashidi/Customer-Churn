-- Total records
SELECT COUNT(*) as total_customers FROM customers;

-- Churn distribution
SELECT 
    Churn, 
    COUNT(*) as count,
    ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM customers), 2) as percentage
FROM customers 
GROUP BY Churn;

-- tenure stats
SELECT 
    MIN(tenure) as min_tenure,
    AVG(tenure) as avg_tenure,
    MAX(tenure) as max_tenure
FROM customers;

-- MonthlyCharges stats
SELECT 
    ROUND(AVG(MonthlyCharges), 2) as avg_monthly,
    ROUND(MIN(MonthlyCharges), 2) as min_monthly,
    ROUND(MAX(MonthlyCharges), 2) as max_monthly
FROM customers;

-- TotalCharges stats
SELECT 
    ROUND(AVG(TotalCharges), 2) as avg_total,
    ROUND(MIN(TotalCharges), 2) as min_total,
    ROUND(MAX(TotalCharges), 2) as max_total
FROM customers;

-- Gender distribution
SELECT gender, COUNT(*) as count FROM customers GROUP BY gender;

-- Contract types
SELECT Contract, COUNT(*) as count FROM customers GROUP BY Contract;

-- Internet Service
SELECT InternetService, COUNT(*) as count FROM customers GROUP BY InternetService;

-- Payment Method
SELECT PaymentMethod, COUNT(*) as count FROM customers GROUP BY PaymentMethod;

-- Churn by Contract
SELECT 
    Contract,
    COUNT(*) as total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate
FROM customers 
GROUP BY Contract;

-- Churn by InternetService
SELECT 
    InternetService,
    COUNT(*) as total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate
FROM customers 
GROUP BY InternetService
ORDER BY churn_rate DESC;

-- Churn by PaymentMethod
SELECT 
    PaymentMethod,
    COUNT(*) as total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate
FROM customers 
GROUP BY PaymentMethod;

-- Churn by tenure groups
SELECT 
    CASE 
        WHEN tenure <= 12 THEN '0-12 months'
        WHEN tenure <= 24 THEN '13-24 months'
        WHEN tenure <= 48 THEN '25-48 months'
        ELSE '49+ months'
    END as tenure_group,
    COUNT(*) as total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate
FROM customers 
GROUP BY tenure_group
ORDER BY tenure_group;

-- Churn by MonthlyCharges ranges
SELECT 
    CASE 
        WHEN MonthlyCharges < 35 THEN 'Low (<$35)'
        WHEN MonthlyCharges < 70 THEN 'Medium ($35-70)'
        WHEN MonthlyCharges < 100 THEN 'High ($70-100)'
        ELSE 'Very High ($100+)'
    END as charge_range,
    COUNT(*) as total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate
FROM customers 
GROUP BY charge_range
ORDER BY charge_range;

-- Churn by SeniorCitizen
SELECT 
    SeniorCitizen,
    COUNT(*) as total,
    SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate
FROM customers 
GROUP BY SeniorCitizen;

-- OnlineSecurity impact
SELECT 
    OnlineSecurity,
    COUNT(*) as total,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate
FROM customers 
GROUP BY OnlineSecurity;

-- TechSupport impact
SELECT 
    TechSupport,
    COUNT(*) as total,
    ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate
FROM customers 
GROUP BY TechSupport;