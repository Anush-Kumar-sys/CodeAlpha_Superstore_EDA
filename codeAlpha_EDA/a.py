import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

data = pd.read_csv("sample - Superstore.csv", encoding="latin1")

print("\n========== DATASET INFORMATION ==========")
print(data.head())
print("\nShape:", data.shape)
print("\nColumns:")
print(data.columns)
print("\nData Types:")
print(data.dtypes)

print("\n========== STATISTICAL SUMMARY ==========")
print(data.describe())

print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())

print("\nMissing Percentage:")
print(data.isnull().mean() * 100)

print("\n========== DUPLICATES ==========")
print("Duplicate rows:", data.duplicated().sum())

data = data.drop_duplicates()

print("Duplicates after cleaning:", data.duplicated().sum())

for column in ["Sales", "Profit", "Discount", "Quantity"]:
    data[column] = pd.to_numeric(data[column], errors="coerce")

data["Order Date"] = pd.to_datetime(data["Order Date"], errors="coerce")
data["Ship Date"] = pd.to_datetime(data["Ship Date"], errors="coerce")

print("\n========== MISSING DATES ==========")
print("Missing Order Dates:", data["Order Date"].isnull().sum())
print("Missing Ship Dates:", data["Ship Date"].isnull().sum())

print("\n========== SALES AND PROFIT ==========")

total_sales = data["Sales"].sum()
total_profit = data["Profit"].sum()

average_sales = data["Sales"].mean()
average_profit = data["Profit"].mean()

print("Total Sales:", total_sales)
print("Total Profit:", total_profit)
print("Average Sales:", average_sales)
print("Average Profit:", average_profit)

print("\n========== CATEGORY ANALYSIS ==========")

category_analysis = data.groupby("Category")[["Sales", "Profit"]].sum()

print(category_analysis)

print("\n========== REGION ANALYSIS ==========")

region_analysis = data.groupby("Region")[["Sales", "Profit"]].sum()

print(region_analysis)

print("\n========== SUB-CATEGORY ANALYSIS ==========")

subcategory_analysis = (
    data.groupby("Sub-Category")[["Sales", "Profit"]]
    .sum()
    .sort_values("Profit", ascending=False)
)

print(subcategory_analysis)

print("\n========== CUSTOMER SEGMENT ==========")

segment_analysis = data.groupby("Segment")[["Sales", "Profit"]].sum()

print(segment_analysis)

print("\n========== TOP 10 PRODUCTS BY SALES ==========")

top_products = (
    data.groupby("Product Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print(top_products)

print("\n========== LOSS-MAKING PRODUCTS ==========")

loss_products = (
    data.groupby("Product Name")["Profit"]
    .sum()
    .sort_values()
    .head(10)
)

print(loss_products)

print("\n========== DISCOUNT VS PROFIT ==========")

correlation = data["Discount"].corr(data["Profit"])

print("Correlation:", correlation)

data["Year"] = data["Order Date"].dt.year

yearly_sales = data.groupby("Year")["Sales"].sum()

print("\n========== YEARLY SALES ==========")
print(yearly_sales)

data["Year-Month"] = data["Order Date"].dt.to_period("M")

monthly_sales = data.groupby("Year-Month")["Sales"].sum()

print("\n========== MONTHLY SALES ==========")
print(monthly_sales)

category_analysis["Sales"].plot(kind="bar", figsize=(8, 5))

plt.title("Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

region_analysis["Profit"].plot(kind="bar", figsize=(8, 5))

plt.title("Profit by Region")
plt.xlabel("Region")
plt.ylabel("Profit")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

yearly_sales.plot(kind="line", marker="o", figsize=(10, 5))

plt.title("Sales Trend by Year")
plt.xlabel("Year")
plt.ylabel("Sales")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

plt.scatter(
    data["Discount"],
    data["Profit"],
    alpha=0.5
)

plt.title("Discount vs Profit")
plt.xlabel("Discount")
plt.ylabel("Profit")
plt.axhline(y=0, linewidth=1)

plt.tight_layout()
plt.show()

segment_analysis["Profit"].plot(kind="bar", figsize=(8, 5))

plt.title("Profit by Customer Segment")
plt.xlabel("Customer Segment")
plt.ylabel("Profit")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

plt.hist(data["Profit"].dropna(), bins=30)

plt.title("Distribution of Profit")
plt.xlabel("Profit")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))

plt.boxplot(data["Profit"].dropna())

plt.title("Profit Boxplot")
plt.ylabel("Profit")

plt.tight_layout()
plt.show()


print("\n========== HYPOTHESIS TEST ==========")


test_data = data[["Discount", "Profit"]].dropna()

r, p_value = pearsonr(
    test_data["Discount"],
    test_data["Profit"]
)

print("Correlation:", r)
print("P-value:", p_value)

alpha = 0.05

if p_value < alpha:
    print("Result: Reject the null hypothesis.")
    print("There is statistical evidence of a linear association between Discount and Profit.")
else:
    print("Result: Fail to reject the null hypothesis.")
    print("There is not sufficient statistical evidence of a linear association between Discount and Profit.")

print("\n========== DATA QUALITY CHECKS ==========")

negative_profit = (data["Profit"] < 0).sum()
negative_sales = (data["Sales"] < 0).sum()
invalid_quantity = (data["Quantity"] <= 0).sum()

print("Negative profit records:", negative_profit)
print("Negative sales records:", negative_sales)
print("Zero/negative quantity records:", invalid_quantity)

print("\nDiscount Statistics:")
print(data["Discount"].describe())

print("\nSales Statistics:")
print(data["Sales"].describe())

print("\nProfit Statistics:")
print(data["Profit"].describe())

print("\n========== EDA COMPLETED ==========")

print("Dataset shape:", data.shape)
print("Total Sales:", total_sales)
print("Total Profit:", total_profit)
print("Discount-Profit Correlation:", correlation)
print("Pearson P-value:", p_value)

print("\nAll EDA analysis and visualizations completed.")