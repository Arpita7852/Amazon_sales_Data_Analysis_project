# AI-Powered Customer Review Analyzer

## Project Description
This project analyzes Amazon product reviews to understand customer sentiment,
ratings, and pricing/discount patterns across product categories. Beyond
standard EDA, it adds two AI layers:
1. **Rule-based sentiment scoring** (VADER) on review text
2. **A trained ML classifier** (TF-IDF + Logistic Regression) that predicts
   sentiment directly from review text

The aggregated findings are also summarized into a plain-English
"Key Insights & Recommendations" section — the kind of deliverable a data
analyst would hand to a business team — and visualized in an interactive
dashboard.

## Dataset
- **Name:** Amazon Sales Dataset
- **Source:** Kaggle — https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset
- **Size:** 1,465 products (1,423 used after cleaning), with pricing,
  ratings, and review text
- **Key columns used:** `product_name`, `category`, `discounted_price`,
  `actual_price`, `discount_percentage`, `rating`, `rating_count`,
  `about_product`, `review_content`

## Technologies Used
- **Python** (Pandas, NumPy) — data cleaning and aggregation
- **Matplotlib** — visualization
- **VADER (vaderSentiment)** — rule-based sentiment analysis
- **scikit-learn** — TF-IDF vectorization + Logistic Regression classifier
- **HTML/Plotly** — interactive results dashboard
- **Claude (LLM)** — AI-generated insights summary from aggregated stats

## Steps Used for Analysis
1. Load and inspect the dataset (`.info()`, `.isnull().sum()`, dtypes)
2. Clean the data — drop 1 corrupted row (shifted columns) and 2 rows with
   missing `rating_count`; strip currency/percent symbols and convert
   price/rating columns to numeric
3. Extract a top-level product category and restrict category comparisons
   to the 3 categories with sufficient sample size (Electronics,
   Computers & Accessories, Home & Kitchen)
4. Run VADER sentiment analysis on `review_content`, classifying each
   product's combined reviews as positive / negative / neutral
5. Aggregate rating, discount %, and sentiment by category
6. Train a TF-IDF + Logistic Regression classifier to predict sentiment
   from review text (with `class_weight='balanced'` to handle the ~96%
   positive / 4% negative class imbalance)
7. Visualize results (sentiment distribution, category comparisons) as
   static charts and as an interactive dashboard
8. Generate an AI-written "Key Insights & Recommendations" summary from
   the aggregated findings

## Key Findings
- Overall sentiment: 95.5% positive, 4.2% negative, 0.3% neutral
  (n = 1,423)
- **Home & Kitchen has the highest negative-sentiment rate (5.8%)**,
  despite receiving the *lowest* average discount (40.2%) — suggesting
  dissatisfaction is more likely tied to product quality than price
- Computers & Accessories rates highest overall (avg. rating 4.16,
  negative rate 3.3%) while carrying the deepest average discount (53.9%)
- ML classifier: 95.8% overall accuracy, but only 42% recall on negative
  reviews — a direct effect of the class imbalance, and a reminder not to
  rely on headline accuracy alone

## AI-Generated Insights
Home & Kitchen products show meaningfully more customer dissatisfaction
than other categories despite smaller discounts, pointing to quality
rather than pricing as the likely driver. Computers & Accessories performs
best on both rating and sentiment and is a strong candidate for continued
promotion. Because both VADER and the trained classifier under-detect
negative sentiment (a known effect of class imbalance), the true
dissatisfaction rate — especially for Home & Kitchen — is likely somewhat
higher than the raw 4.2% figure suggests.

## Interactive Dashboard
A live results dashboard (sentiment breakdown, category comparisons, and
ML performance) is available here:
https://claude.ai/artifact/14eSpJ41sAEDcKC2vkjvyA

## Setup / Run Instructions
1. Download the dataset CSV from the Kaggle link above
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Place the CSV in the same folder as the notebook
4. Open the notebook:
   ```
   jupyter notebook project_code.ipynb
   ```
5. Run all cells in order (Kernel → Restart & Run All)

## Author
Arpita Sethi — Data Analyst (Fresher), AICTE–IBM SkillsBuild Internship
(BharatCares)
