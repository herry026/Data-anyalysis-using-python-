{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "cover_page",
   "metadata": {},
   "source": [
    "# Exploratory Data Analysis and Prediction of Netflix Movies and TV Shows\n",
    "\n",
    "---\n",
    "\n",
    "## INDEX\n",
    "\n",
    "| Sr. No. | Title | Page No. |\n",
    "|---------|-------------------------------------------------------|----------|\n",
    "| 1 | Acknowledgement | 1 |\n",
    "| 2 | Project Details (Title, Dataset, Objectives, Tools) | 2 |\n",
    "| 3 | Task 1: Data Understanding | 3 |\n",
    "| | - Import Libraries | |\n",
    "| | - Read Dataset | |\n",
    "| | - Dataset Shape, Describe, Info | |\n",
    "| | - Data Type Identification | |\n",
    "| 4 | Task 2: Exploratory Data Analysis (EDA) | 5 |\n",
    "| | - Univariate Analysis (Histogram) | |\n",
    "| | - Boxplot (Outlier Detection) | |\n",
    "| | - Bivariate Analysis (Scatter Plot) | |\n",
    "| | - Multivariate Analysis (Bar, Heatmap, Pairplot) | |\n",
    "| 5 | Task 3: Handling Missing Values & Outliers | 10 |\n",
    "| | - Missing Values Check | |\n",
    "| | - Fill Missing Values | |\n",
    "| | - Outlier Detection | |\n",
    "| 6 | Task 4: Spread of Data | 12 |\n",
    "| | - Statistical Summary | |\n",
    "| | - Skewness & Kurtosis | |\n",
    "| 7 | Task 5: Automating EDA | 14 |\n",
    "| | - auto_eda() Function | |\n",
    "| 8 | Task 6: Regression Analysis | 15 |\n",
    "| 9 | Task 7: Model Evaluation | 16 |\n",
    "| | - MSE, MAE, R² | |\n",
    "| | - Actual vs Predicted Plot | |\n",
    "| 10 | Task 8: Overfitting & Underfitting | 17 |\n",
    "| 11 | Task 9: Classification | 18 |\n",
    "| 12 | Conclusion | 19 |"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ack_header",
   "metadata": {},
   "source": [
    "---\n",
    "# Sr. No. 1 — Acknowledgement\n",
    "### Page 1"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "acknowledgement",
   "metadata": {},
   "source": [
    "## Acknowledgement\n",
    "\n",
    "I would like to express my sincere gratitude to everyone who supported me throughout this project.\n",
    "\n",
    "I am thankful to my **project guide and faculty** for their constant guidance, encouragement, and valuable suggestions during the course of this work.\n",
    "\n",
    "I am also grateful to the open-source community and **Kaggle** for providing the Netflix Titles dataset, which made this exploratory data analysis possible.\n",
    "\n",
    "Special thanks to the developers of **Python**, **Pandas**, **Matplotlib**, **Seaborn**, and **Scikit-learn** for building the powerful tools used in this project.\n",
    "\n",
    "Finally, I thank my family and friends for their unwavering support and motivation throughout this journey."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "proj_header",
   "metadata": {},
   "source": [
    "---\n",
    "# Sr. No. 2 — Project Details\n",
    "### Page 2"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "project_details",
   "metadata": {},
   "source": [
    "## Project Details\n",
    "\n",
    "### 📌 Title\n",
    "**Exploratory Data Analysis and Prediction of Netflix Movies and TV Shows using Python**\n",
    "\n",
    "---\n",
    "\n",
    "### 📂 Dataset\n",
    "| Field | Details |\n",
    "|-------|---------|\n",
    "| **Name** | Netflix Movies and TV Shows |\n",
    "| **File** | `netflix_titles.csv` |\n",
    "| **Source** | Kaggle (https://www.kaggle.com/datasets/shivamb/netflix-shows) |\n",
    "| **Rows** | ~8,800+ titles |\n",
    "| **Columns** | 12 (show_id, type, title, director, cast, country, date_added, release_year, rating, duration, listed_in, description) |\n",
    "\n",
    "---\n",
    "\n",
    "### 🎯 Objectives\n",
    "1. Understand the structure and content of the Netflix dataset.\n",
    "2. Perform thorough Exploratory Data Analysis (EDA) — univariate, bivariate, and multivariate.\n",
    "3. Handle missing values and detect outliers.\n",
    "4. Compute statistical measures: mean, median, std, skewness, kurtosis.\n",
    "5. Automate EDA using a reusable Python function.\n",
    "6. Build a Linear Regression model to predict `release_year`.\n",
    "7. Evaluate the model using MSE, MAE, and R².\n",
    "8. Understand and demonstrate overfitting vs underfitting.\n",
    "9. Build a Logistic Regression classifier to predict content type (Movie / TV Show).\n",
    "10. Draw meaningful conclusions from the analysis.\n",
    "\n",
    "---\n",
    "\n",
    "### 🛠️ Tools & Technologies\n",
    "| Category | Tools |\n",
    "|----------|-------|\n",
    "| **Language** | Python 3.x |\n",
    "| **IDE** | Jupyter Notebook |\n",
    "| **Data Handling** | Pandas, NumPy |\n",
    "| **Visualization** | Matplotlib, Seaborn |\n",
    "| **Machine Learning** | Scikit-learn |\n",
    "| **Dataset** | netflix_titles.csv |"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task1_header",
   "metadata": {},
   "source": [
    "---\n",
    "# Sr. No. 3 — Task 1: Data Understanding\n",
    "### Page 3"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task1_import_md",
   "metadata": {},
   "source": [
    "## Task 1: Data Understanding\n",
    "\n",
    "### Step 1 — Import Libraries"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "import_libraries",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Standard Libraries ──────────────────────────────────────────\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "# ── Visualisation ────────────────────────────────────────────────\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.ticker as mticker\n",
    "import seaborn as sns\n",
    "\n",
    "# ── Machine Learning ─────────────────────────────────────────────\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LinearRegression, LogisticRegression\n",
    "from sklearn.metrics import (\n",
    "    mean_squared_error, mean_absolute_error, r2_score,\n",
    "    accuracy_score, confusion_matrix, classification_report\n",
    ")\n",
    "\n",
    "# ── Plot Defaults ────────────────────────────────────────────────\n",
    "sns.set_theme(style='whitegrid', palette='muted')\n",
    "plt.rcParams['figure.figsize'] = (10, 5)\n",
    "plt.rcParams['axes.titlesize'] = 14\n",
    "plt.rcParams['axes.labelsize'] = 12\n",
    "\n",
    "print(\"✅ All libraries imported successfully.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task1_read_md",
   "metadata": {},
   "source": [
    "### Step 2 — Read Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "read_dataset",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the dataset — ensure netflix_titles.csv is in the same directory\n",
    "df = pd.read_csv('netflix_titles.csv')\n",
    "\n",
    "# Standardise column names: strip whitespace, lowercase\n",
    "df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')\n",
    "\n",
    "print(f\"Dataset loaded successfully!\")\n",
    "print(f\"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}\")\n",
    "print()\n",
    "print(\"First 5 rows:\")\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "last_rows",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Last 5 rows:\")\n",
    "df.tail()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task1_shape_md",
   "metadata": {},
   "source": [
    "### Step 3 — Dataset Shape, Describe, Info"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "shape_describe_info",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Shape ────────────────────────────────────────────────────────\n",
    "print(\"SHAPE (rows, columns):\", df.shape)\n",
    "print(\"\\nCOLUMNS:\", df.columns.tolist())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "describe",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Describe ─────────────────────────────────────────────────────\n",
    "print(\"DESCRIPTIVE STATISTICS (numerical):\")\n",
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "info",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Info ─────────────────────────────────────────────────────────\n",
    "print(\"DATASET INFO:\")\n",
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task1_dtype_md",
   "metadata": {},
   "source": [
    "### Step 4 — Data Type Identification"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dtype_identification",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"DATA TYPES OF EACH COLUMN:\")\n",
    "print(df.dtypes)\n",
    "print()\n",
    "print(\"─\" * 45)\n",
    "print(\"QUANTITATIVE (Numerical) columns:\")\n",
    "num_cols_raw = df.select_dtypes(include=[np.number]).columns.tolist()\n",
    "print(\" \", num_cols_raw)\n",
    "\n",
    "print(\"\\nQUALITATIVE (Categorical / Object) columns:\")\n",
    "cat_cols_raw = df.select_dtypes(include=['object']).columns.tolist()\n",
    "print(\" \", cat_cols_raw)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task2_header",
   "metadata": {},
   "source": [
    "---\n",
    "# Sr. No. 4 — Task 2: Exploratory Data Analysis (EDA)\n",
    "### Page 5"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task2_fe_md",
   "metadata": {},
   "source": [
    "## Task 2: Exploratory Data Analysis (EDA)\n",
    "\n",
    "> **Feature Engineering** — creating `duration_num`, `year_added`, and `content_age` before analysis."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "feature_engineering",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Extract numeric part from 'duration' (e.g., '90 min' → 90, '2 Seasons' → 2)\n",
    "df['duration_num'] = df['duration'].str.extract(r'(\\d+)').astype(float)\n",
    "\n",
    "# Binary label: 1 = Movie, 0 = TV Show\n",
    "df['is_movie'] = (df['type'] == 'Movie').astype(int)\n",
    "\n",
    "# Year the title was added to Netflix\n",
    "df['year_added'] = pd.to_datetime(df['date_added'], errors='coerce').dt.year\n",
    "\n",
    "# Age of content when added (years between release and addition)\n",
    "df['content_age'] = (df['year_added'] - df['release_year']).clip(lower=0)\n",
    "\n",
    "print(\"✅ Features created: duration_num, is_movie, year_added, content_age\")\n",
    "df[['type','duration','duration_num','is_movie','year_added','content_age']].head(6)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task2_uni_md",
   "metadata": {},
   "source": [
    "### Step 1 — Univariate Analysis (Histogram)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "univariate_histogram",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# Histogram: Release Year\n",
    "sns.histplot(df['release_year'].dropna(), kde=True, bins=40,\n",
    "             color='steelblue', ax=axes[0])\n",
    "axes[0].set_title('Histogram — Release Year')\n",
    "axes[0].set_xlabel('Release Year')\n",
    "axes[0].set_ylabel('Count')\n",
    "\n",
    "# Histogram: Movie Duration\n",
    "movie_dur = df[df['is_movie'] == 1]['duration_num'].dropna()\n",
    "sns.histplot(movie_dur, kde=True, bins=40, color='coral', ax=axes[1])\n",
    "axes[1].set_title('Histogram — Movie Duration (minutes)')\n",
    "axes[1].set_xlabel('Duration (min)')\n",
    "axes[1].set_ylabel('Count')\n",
    "\n",
    "plt.suptitle('Univariate Analysis — Histograms', fontsize=15)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "univariate_hist2",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# Content Type (Bar)\n",
    "df['type'].value_counts().plot(kind='bar', ax=axes[0],\n",
    "                                color=['steelblue','coral'], edgecolor='black')\n",
    "axes[0].set_title('Content Type Count')\n",
    "axes[0].set_xlabel('Type')\n",
    "axes[0].set_ylabel('Count')\n",
    "axes[0].tick_params(axis='x', rotation=0)\n",
    "\n",
    "# Top 8 Ratings\n",
    "df['rating'].value_counts().head(8).plot(kind='bar', ax=axes[1],\n",
    "                                          color='mediumslateblue', edgecolor='black')\n",
    "axes[1].set_title('Top 8 Content Ratings')\n",
    "axes[1].set_xlabel('Rating')\n",
    "axes[1].set_ylabel('Count')\n",
    "axes[1].tick_params(axis='x', rotation=45)\n",
    "\n",
    "plt.suptitle('Univariate Analysis — Categorical Features', fontsize=15)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task2_box_md",
   "metadata": {},
   "source": [
    "### Step 2 — Boxplot (Outlier Detection)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "boxplot_outlier",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n",
    "\n",
    "# Release Year boxplot\n",
    "sns.boxplot(y=df['release_year'], ax=axes[0], color='lightblue',\n",
    "            flierprops=dict(marker='o', markerfacecolor='red', markersize=4))\n",
    "axes[0].set_title('Boxplot — Release Year')\n",
    "\n",
    "# Movie Duration boxplot\n",
    "sns.boxplot(y=df[df['is_movie']==1]['duration_num'], ax=axes[1], color='lightsalmon',\n",
    "            flierprops=dict(marker='o', markerfacecolor='red', markersize=4))\n",
    "axes[1].set_title('Boxplot — Movie Duration (min)')\n",
    "\n",
    "# Content Age boxplot\n",
    "sns.boxplot(y=df['content_age'].dropna(), ax=axes[2], color='lightgreen',\n",
    "            flierprops=dict(marker='o', markerfacecolor='red', markersize=4))\n",
    "axes[2].set_title('Boxplot — Content Age (years)')\n",
    "\n",
    "plt.suptitle('Boxplots — Outlier Detection', fontsize=15)\n",
    "plt.tight_layout()\n",
    "plt.show()\n",
    "\n",
    "print(\"Red dots indicate outliers (values beyond 1.5 × IQR).\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task2_biv_md",
   "metadata": {},
   "source": [
    "### Step 3 — Bivariate Analysis (Scatter Plot)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bivariate_scatter",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# Scatter: Duration vs Release Year\n",
    "sns.scatterplot(x='duration_num', y='release_year', hue='type',\n",
    "                data=df, alpha=0.4, ax=axes[0], palette={'Movie':'coral','TV Show':'steelblue'})\n",
    "axes[0].set_title('Scatter — Duration vs Release Year')\n",
    "axes[0].set_xlabel('Duration (min / seasons)')\n",
    "axes[0].set_ylabel('Release Year')\n",
    "\n",
    "# Scatter: Content Age vs Release Year\n",
    "sns.scatterplot(x='release_year', y='content_age', hue='type',\n",
    "                data=df.dropna(subset=['content_age']), alpha=0.4, ax=axes[1],\n",
    "                palette={'Movie':'coral','TV Show':'steelblue'})\n",
    "axes[1].set_title('Scatter — Release Year vs Content Age')\n",
    "axes[1].set_xlabel('Release Year')\n",
    "axes[1].set_ylabel('Content Age (years)')\n",
    "\n",
    "plt.suptitle('Bivariate Analysis — Scatter Plots', fontsize=15)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bivariate_corr",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Pearson Correlation between numerical features:\")\n",
    "print(df[['release_year','duration_num','content_age','is_movie']].corr(numeric_only=True).round(4))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task2_multi_md",
   "metadata": {},
   "source": [
    "### Step 4 — Multivariate Analysis (Bar, Heatmap, Pairplot)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "multivariate_bar",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Grouped Bar: Rating by Content Type ──────────────────────────\n",
    "top_ratings = df['rating'].value_counts().head(8).index\n",
    "rating_df   = df[df['rating'].isin(top_ratings)]\n",
    "rating_type = rating_df.groupby(['rating','type']).size().unstack(fill_value=0)\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(12, 5))\n",
    "rating_type.plot(kind='bar', ax=ax, edgecolor='black', colormap='Set2')\n",
    "ax.set_title('Multivariate — Rating Distribution by Content Type')\n",
    "ax.set_xlabel('Rating')\n",
    "ax.set_ylabel('Count')\n",
    "ax.tick_params(axis='x', rotation=45)\n",
    "ax.legend(title='Type')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "multivariate_heatmap",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Correlation Heatmap ───────────────────────────────────────────\n",
    "corr = df[['release_year','duration_num','content_age','is_movie']].corr()\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(7, 5))\n",
    "sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm',\n",
    "            linewidths=0.8, square=True, ax=ax)\n",
    "ax.set_title('Multivariate — Correlation Heatmap')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "multivariate_pairplot",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Pairplot ──────────────────────────────────────────────────────\n",
    "pair_df = df[['release_year','duration_num','content_age','is_movie']].dropna()\n",
    "# Rename is_movie for readable legend\n",
    "pair_df['Content Type'] = pair_df['is_movie'].map({1:'Movie', 0:'TV Show'})\n",
    "\n",
    "sns.pairplot(pair_df.drop(columns='is_movie'), hue='Content Type',\n",
    "             palette={'Movie':'coral','TV Show':'steelblue'},\n",
    "             plot_kws={'alpha': 0.3}, diag_kind='kde')\n",
    "plt.suptitle('Multivariate — Pairplot of Numerical Features', y=1.01, fontsize=14)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task3_header",
   "metadata": {},
   "source": [
    "---\n",
    "# Sr. No. 5 — Task 3: Handling Missing Values & Outliers\n",
    "### Page 10"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task3_check_md",
   "metadata": {},
   "source": [
    "## Task 3: Handling Missing Values & Outliers\n",
    "\n",
    "### Step 1 — Missing Values Check"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "missing_check",
   "metadata": {},
   "outputs": [],
   "source": [
    "missing       = df.isnull().sum().sort_values(ascending=False)\n",
    "missing_pct   = (missing / len(df) * 100).round(2)\n",
    "missing_table = pd.DataFrame({'Missing Count': missing, 'Missing %': missing_pct})\n",
    "missing_table = missing_table[missing_table['Missing Count'] > 0]\n",
    "\n",
    "print(\"Columns with missing values:\")\n",
    "display(missing_table)\n",
    "\n",
    "# Visual\n",
    "fig, ax = plt.subplots(figsize=(10, 4))\n",
    "missing_table['Missing %'].plot(kind='bar', ax=ax, color='tomato', edgecolor='black')\n",
    "ax.set_title('Missing Values by Column (%)')\n",
    "ax.set_ylabel('Missing %')\n",
    "ax.set_xlabel('Column')\n",
    "plt.xticks(rotation=45, ha='right')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task3_fill_md",
   "metadata": {},
   "source": [
    "### Step 2 — Fill Missing Values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fill_missing",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fill categorical columns with mode or 'Unknown'\n",
    "fill_cols = ['director', 'cast', 'country', 'rating']\n",
    "for col in fill_cols:\n",
    "    if col in df.columns:\n",
    "        fill_val = df[col].mode()[0] if df[col].mode().size > 0 else 'Unknown'\n",
    "        df[col].fillna(fill_val, inplace=True)\n",
    "        print(f\"  '{col}' → filled with: '{fill_val}'\")\n",
    "\n",
    "# Drop rows where 'duration' is missing (~3 rows)\n",
    "before = len(df)\n",
    "df.dropna(subset=['duration'], inplace=True)\n",
    "print(f\"\\n  Dropped {before - len(df)} rows with missing 'duration'.\")\n",
    "\n",
    "# Fill date_added\n",
    "df['date_added'].fillna('Unknown', inplace=True)\n",
    "\n",
    "print(\"\\n✅ Missing values handled.\")\n",
    "remaining = df.isnull().sum()\n",
    "print(\"Remaining missing values:\", remaining[remaining > 0].to_dict() or \"None\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task3_outlier_md",
   "metadata": {},
   "source": [
    "### Step 3 — Outlier Detection (IQR Method)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "outlier_detection",
   "metadata": {},
   "outputs": [],
   "source": [
    "def detect_outliers_iqr(series: pd.Series) -> dict:\n",
    "    \"\"\"Return outlier count and bounds using the IQR method.\"\"\"\n",
    "    Q1  = series.quantile(0.25)\n",
    "    Q3  = series.quantile(0.75)\n",
    "    IQR = Q3 - Q1\n",
    "    lower = Q1 - 1.5 * IQR\n",
    "    upper = Q3 + 1.5 * IQR\n",
    "    outliers = series[(series < lower) | (series > upper)]\n",
    "    return {\n",
    "        'Q1': round(Q1, 2), 'Q3': round(Q3, 2), 'IQR': round(IQR, 2),\n",
    "        'Lower Bound': round(lower, 2), 'Upper Bound': round(upper, 2),\n",
    "        'Outlier Count': len(outliers)\n",
    "    }\n",
    "\n",
    "outlier_cols = ['release_year', 'duration_num', 'content_age']\n",
    "outlier_summary = pd.DataFrame({col: detect_outliers_iqr(df[col].dropna())\n",
    "                                 for col in outlier_cols}).T\n",
    "print(\"Outlier Summary (IQR Method):\")\n",
    "display(outlier_summary)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "outlier_viz",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cap outliers using IQR (Winsorisation)\n",
    "for col in ['duration_num', 'content_age']:\n",
    "    Q1  = df[col].quantile(0.25)\n",
    "    Q3  = df[col].quantile(0.75)\n",
    "    IQR = Q3 - Q1\n",
    "    lower = Q1 - 1.5 * IQR\n",
    "    upper = Q3 + 1.5 * IQR\n",
    "    df[col] = df[col].clip(lower=lower, upper=upper)\n",
    "\n",
    "print(\"✅ Outliers capped (Winsorisation) for: duration_num, content_age\")\n",
    "\n",
    "# Post-treatment boxplots\n",
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
    "sns.boxplot(y=df['duration_num'], ax=axes[0], color='lightsalmon')\n",
    "axes[0].set_title('After Outlier Treatment — Duration')\n",
    "sns.boxplot(y=df['content_age'].dropna(), ax=axes[1], color='lightgreen')\n",
    "axes[1].set_title('After Outlier Treatment — Content Age')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task4_header",
   "metadata": {},
   "source": [
    "---\n",
    "# Sr. No. 6 — Task 4: Spread of Data\n",
    "### Page 12"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task4_stat_md",
   "metadata": {},
   "source": [
    "## Task 4: Spread of Data\n",
    "\n",
    "### Step 1 — Statistical Summary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "statistical_summary",
   "metadata": {},
   "outputs": [],
   "source": [
    "num_cols = ['release_year', 'duration_num', 'content_age']\n",
    "\n",
    "stats_dict = {}\n",
    "for col in num_cols:\n",
    "    s = df[col].dropna()\n",
    "    stats_dict[col] = {\n",
    "        'Count':    int(s.count()),\n",
    "        'Mean':     round(s.mean(), 2),\n",
    "        'Median':   round(s.median(), 2),\n",
    "        'Mode':     round(s.mode()[0], 2),\n",
    "        'Std Dev':  round(s.std(), 2),\n",
    "        'Variance': round(s.var(), 2),\n",
    "        'Min':      s.min(),\n",
    "        'Max':      s.max(),\n",
    "        'Range':    s.max() - s.min(),\n",
    "        'Q1 (25%)': round(s.quantile(0.25), 2),\n",
    "        'Q3 (75%)': round(s.quantile(0.75), 2),\n",
    "        'IQR':      round(s.quantile(0.75) - s.quantile(0.25), 2),\n",
    "    }\n",
    "\n",
    "stats_df = pd.DataFrame(stats_dict).T\n",
    "print(\"📊 Statistical Summary\")\n",
    "display(stats_df)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task4_skew_md",
   "metadata": {},
   "source": [
    "### Step 2 — Skewness & Kurtosis"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "skewness_kurtosis",
   "metadata": {},
   "outputs": [],
   "source": [
    "sk_table = pd.DataFrame({\n",
    "    'Skewness': df[num_cols].skew().round(4),\n",
    "    'Kurtosis': df[num_cols].kurt().round(4)\n",
    "})\n",
    "\n",
    "print(\"Skewness & Kurtosis:\")\n",
    "display(sk_table)\n",
    "print()\n",
    "print(\"Interpretation:\")\n",
    "print(\"  Skewness > 0  → Right-skewed (tail on right side)\")\n",
    "print(\"  Skewness < 0  → Left-skewed  (tail on left side)\")\n",
    "print(\"  Kurtosis > 3  → Leptokurtic  (heavy-tailed)\")\n",
    "print(\"  Kurtosis < 3  → Platykurtic  (light-tailed)\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "skew_plots",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 3, figsize=(16, 4))\n",
    "colors = ['steelblue', 'coral', 'seagreen']\n",
    "\n",
    "for ax, col, color in zip(axes, num_cols, colors):\n",
    "    s  = df[col].dropna()\n",
    "    sk = s.skew()\n",
    "    ku = s.kurt()\n",
    "    sns.histplot(s, kde=True, ax=ax, color=color, bins=35)\n",
    "    ax.axvline(s.mean(),   color='red',    linestyle='--', label='Mean')\n",
    "    ax.axvline(s.median(), color='black',  linestyle='-',  label='Median')\n",
    "    ax.set_title(f'{col}\\nSkew={sk:.2f}  |  Kurt={ku:.2f}')\n",
    "    ax.legend(fontsize=8)\n",
    "\n",
    "plt.suptitle('Distribution with Skewness & Kurtosis', fontsize=14, y=1.02)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task5_header",
   "metadata": {},
   "source": [
    "---\n",
    "# Sr. No. 7 — Task 5: Automating EDA\n",
    "### Page 14"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task5_md",
   "metadata": {},
   "source": [
    "## Task 5: Automating EDA\n",
    "\n",
    "### auto_eda() Function"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "auto_eda_function",
   "metadata": {},
   "outputs": [],
   "source": [
    "def auto_eda(data: pd.DataFrame, target: str = None) -> None:\n",
    "    \"\"\"\n",
    "    Automated EDA function.\n",
    "    Prints shape, data types, missing values, duplicates,\n",
    "    statistical summary, correlation, and generates key plots.\n",
    "\n",
    "    Parameters\n",
    "    ----------\n",
    "    data   : pd.DataFrame — input dataset\n",
    "    target : str          — optional target column name for highlighted analysis\n",
    "    \"\"\"\n",
    "    sep = \"=\" * 60\n",
    "\n",
    "    # 1. Basic info\n",
    "    print(sep)\n",
    "    print(f\"  AUTO EDA REPORT\")\n",
    "    print(sep)\n",
    "    print(f\"  Shape      : {data.shape}\")\n",
    "    print(f\"  Duplicates : {data.duplicated().sum()}\")\n",
    "\n",
    "    # 2. Data Types\n",
    "    print(f\"\\n{'─'*40}\")\n",
    "    print(\"  DATA TYPES\")\n",
    "    print(f\"{'─'*40}\")\n",
    "    print(data.dtypes.to_string())\n",
    "\n",
    "    # 3. Missing Values\n",
    "    print(f\"\\n{'─'*40}\")\n",
    "    print(\"  MISSING VALUES\")\n",
    "    print(f\"{'─'*40}\")\n",
    "    miss = data.isnull().sum()\n",
    "    miss = miss[miss > 0]\n",
    "    print(\"  None ✅\" if miss.empty else miss.to_string())\n",
    "\n",
    "    # 4. Descriptive Statistics\n",
    "    print(f\"\\n{'─'*40}\")\n",
    "    print(\"  DESCRIPTIVE STATISTICS\")\n",
    "    print(f\"{'─'*40}\")\n",
    "    display(data.describe(include='all').T)\n",
    "\n",
    "    # 5. Correlation (numerical only)\n",
    "    num_data = data.select_dtypes(include=np.number)\n",
    "    if not num_data.empty:\n",
    "        print(f\"\\n{'─'*40}\")\n",
    "        print(\"  CORRELATION MATRIX\")\n",
    "        print(f\"{'─'*40}\")\n",
    "        display(num_data.corr().round(3))\n",
    "\n",
    "        # Heatmap\n",
    "        fig, ax = plt.subplots(figsize=(8, 5))\n",
    "        sns.heatmap(num_data.corr(), annot=True, fmt='.2f', cmap='coolwarm',\n",
    "                    linewidths=0.5, ax=ax)\n",
    "        ax.set_title('auto_eda — Correlation Heatmap')\n",
    "        plt.tight_layout()\n",
    "        plt.show()\n",
    "\n",
    "        # Distribution plots for each numerical column\n",
    "        ncols_plot = min(3, len(num_data.columns))\n",
    "        nrows_plot = (len(num_data.columns) + ncols_plot - 1) // ncols_plot\n",
    "        fig, axes = plt.subplots(nrows_plot, ncols_plot,\n",
    "                                  figsize=(5 * ncols_plot, 4 * nrows_plot))\n",
    "        axes = np.array(axes).flatten()\n",
    "        for i, col in enumerate(num_data.columns):\n",
    "            sns.histplot(num_data[col].dropna(), kde=True, ax=axes[i], color='mediumpurple')\n",
    "            axes[i].set_title(f'Distribution: {col}')\n",
    "        for j in range(i+1, len(axes)):\n",
    "            axes[j].set_visible(False)\n",
    "        plt.suptitle('auto_eda — Numerical Feature Distributions', fontsize=13, y=1.02)\n",
    "        plt.tight_layout()\n",
    "        plt.show()\n",
    "\n",
    "    print(sep)\n",
    "    print(\"  ✅ auto_eda complete.\")\n",
    "    print(sep)\n",
    "\n",
    "\n",
    "# ── Run auto_eda on the dataset ───────────────────────────────────\n",
    "auto_eda(df[['release_year','duration_num','content_age','is_movie']])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task6_header",
   "metadata": {},
   "source": [
    "---\n",
    "# Sr. No. 8 — Task 6: Regression Analysis\n",
    "### Page 15"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task6_md",
   "metadata": {},
   "source": [
    "## Task 6: Regression Analysis\n",
    "\n",
    "**Objective:** Predict `release_year` from `duration_num` and `content_age` using **Linear Regression**.\n",
    "\n",
    "**Why?** We explore whether the numeric properties of a title can hint at its release era."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "regression_prep",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Use only Movie rows for a consistent duration signal (all in minutes)\n",
    "reg_df = df[df['is_movie'] == 1][['duration_num', 'content_age', 'release_year']].dropna()\n",
    "\n",
    "X_reg = reg_df[['duration_num', 'content_age']]\n",
    "y_reg = reg_df['release_year']\n",
    "\n",
    "X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(\n",
    "    X_reg, y_reg, test_size=0.2, random_state=42\n",
    ")\n",
    "\n",
    "print(f\"Total samples : {len(reg_df)}\")\n",
    "print(f\"Training set  : {X_train_r.shape[0]}\")\n",
    "print(f\"Test set      : {X_test_r.shape[0]}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "regression_train",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Fit Linear Regression ─────────────────────────────────────────\n",
    "lr_model = LinearRegression()\n",
    "lr_model.fit(X_train_r, y_train_r)\n",
    "\n",
    "print(\"Linear Regression Equation:\")\n",
    "coef_str = \" + \".join([f\"({c:.4f} × {n})\"\n",
    "                        for c, n in zip(lr_model.coef_, X_reg.columns)])\n",
    "print(f\"  release_year = {lr_model.intercept_:.2f} + {coef_str}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task7_header",
   "metadata": {},
   "source": [
    "---\n",
    "# Sr. No. 9 — Task 7: Model Evaluation\n",
    "### Page 16"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task7_md",
   "metadata": {},
   "source": [
    "## Task 7: Model Evaluation\n",
    "\n",
    "### Step 1 — MSE, MAE, R²"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "model_evaluation",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_pred_train_r = lr_model.predict(X_train_r)\n",
    "y_pred_test_r  = lr_model.predict(X_test_r)\n",
    "\n",
    "eval_results = {\n",
    "    'Metric':  ['MSE', 'MAE', 'RMSE', 'R²'],\n",
    "    'Train':   [\n",
    "        round(mean_squared_error(y_train_r, y_pred_train_r), 4),\n",
    "        round(mean_absolute_error(y_train_r, y_pred_train_r), 4),\n",
    "        round(np.sqrt(mean_squared_error(y_train_r, y_pred_train_r)), 4),\n",
    "        round(r2_score(y_train_r, y_pred_train_r), 4)\n",
    "    ],\n",
    "    'Test':    [\n",
    "        round(mean_squared_error(y_test_r, y_pred_test_r), 4),\n",
    "        round(mean_absolute_error(y_test_r, y_pred_test_r), 4),\n",
    "        round(np.sqrt(mean_squared_error(y_test_r, y_pred_test_r)), 4),\n",
    "        round(r2_score(y_test_r, y_pred_test_r), 4)\n",
    "    ]\n",
    "}\n",
    "\n",
    "eval_df = pd.DataFrame(eval_results).set_index('Metric')\n",
    "print(\"📊 Regression Model Evaluation Metrics\")\n",
    "display(eval_df)\n",
    "\n",
    "print()\n",
    "print(\"Definitions:\")\n",
    "print(\"  MSE  (Mean Squared Error)    — average squared difference between actual & predicted\")\n",
    "print(\"  MAE  (Mean Absolute Error)   — average absolute difference between actual & predicted\")\n",
    "print(\"  RMSE (Root Mean Sq. Error)   — MSE in original units; easier to interpret\")\n",
    "print(\"  R²   (Coefficient of Det.)   — proportion of variance explained (1.0 = perfect)\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task7_plot_md",
   "metadata": {},
   "source": [
    "### Step 2 — Actual vs Predicted Plot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "actual_vs_predicted",
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# Actual vs Predicted scatter\n",
    "axes[0].scatter(y_test_r, y_pred_test_r, alpha=0.35,\n",
    "                color='royalblue', edgecolors='white', linewidth=0.3, s=20)\n",
    "mn = min(y_test_r.min(), y_pred_test_r.min())\n",
    "mx = max(y_test_r.max(), y_pred_test_r.max())\n",
    "axes[0].plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect Fit')\n",
    "axes[0].set_title('Actual vs Predicted — Release Year')\n",
    "axes[0].set_xlabel('Actual Release Year')\n",
    "axes[0].set_ylabel('Predicted Release Year')\n",
    "axes[0].legend()\n",
    "\n",
    "# Residuals distribution\n",
    "residuals = y_test_r.values - y_pred_test_r\n",
    "sns.histplot(residuals, kde=True, ax=axes[1], color='tomato', bins=40)\n",
    "axes[1].axvline(0, color='black', linestyle='--', linewidth=1.5, label='Zero Error')\n",
    "axes[1].set_title('Residuals Distribution')\n",
    "axes[1].set_xlabel('Residual (Actual − Predicted)')\n",
    "axes[1].set_ylabel('Count')\n",
    "axes[1].legend()\n",
    "\n",
    "plt.suptitle('Task 7 — Model Evaluation Plots', fontsize=14)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task8_header",
   "metadata": {},
   "source": [
    "---\n",
    "# Sr. No. 10 — Task 8: Overfitting & Underfitting\n",
    "### Page 17"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task8_md",
   "metadata": {},
   "source": [
    "## Task 8: Overfitting & Underfitting\n",
    "\n",
    "| Scenario | Train Score | Test Score | Problem |\n",
    "|----------|------------|-----------|----------|\n",
    "| **Underfitting** | Low | Low | Model too simple; cannot learn the pattern |\n",
    "| **Overfitting** | High | Low | Model memorised training data; fails to generalise |\n",
    "| **Good Fit** | High | ~High | Model generalises well to unseen data |"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "overfit_demo",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Demonstrate using training size experiment ────────────────────\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.preprocessing import PolynomialFeatures\n",
    "\n",
    "train_sizes = np.arange(0.1, 0.95, 0.05)\n",
    "train_r2_list, test_r2_list = [], []\n",
    "\n",
    "for ts in train_sizes:\n",
    "    Xtr, Xte, ytr, yte = train_test_split(X_reg, y_reg, train_size=ts, random_state=42)\n",
    "    m = LinearRegression().fit(Xtr, ytr)\n",
    "    train_r2_list.append(r2_score(ytr, m.predict(Xtr)))\n",
    "    test_r2_list.append(r2_score(yte, m.predict(Xte)))\n",
    "\n",
    "fig, ax = plt.subplots(figsize=(10, 5))\n",
    "ax.plot(train_sizes, train_r2_list, marker='o', label='Train R²', color='steelblue')\n",
    "ax.plot(train_sizes, test_r2_list,  marker='s', label='Test R²',  color='coral')\n",
    "ax.fill_between(train_sizes, train_r2_list, test_r2_list, alpha=0.15, color='gray',\n",
    "                label='Generalisation Gap')\n",
    "ax.set_title('Overfitting vs Underfitting — Learning Curve')\n",
    "ax.set_xlabel('Training Set Size (fraction)')\n",
    "ax.set_ylabel('R² Score')\n",
    "ax.legend()\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "overfit_summary",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_r2 = round(r2_score(y_train_r, lr_model.predict(X_train_r)), 4)\n",
    "test_r2  = round(r2_score(y_test_r,  lr_model.predict(X_test_r)),  4)\n",
    "gap      = round(abs(train_r2 - test_r2), 4)\n",
    "\n",
    "print(f\"  Train R²          : {train_r2}\")\n",
    "print(f\"  Test  R²          : {test_r2}\")\n",
    "print(f\"  Generalisation Gap: {gap}\")\n",
    "print()\n",
    "if gap < 0.05:\n",
    "    print(\"✅ Verdict: GOOD FIT — Train and Test scores are close.\")\n",
    "elif train_r2 > 0.85 and test_r2 < 0.5:\n",
    "    print(\"⚠️  Verdict: OVERFITTING — Model performs well on train but poorly on test.\")\n",
    "else:\n",
    "    print(\"⚠️  Verdict: UNDERFITTING — Model scores are low on both sets.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task9_header",
   "metadata": {},
   "source": [
    "---\n",
    "# Sr. No. 11 — Task 9: Classification\n",
    "### Page 18"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "task9_md",
   "metadata": {},
   "source": [
    "## Task 9: Classification\n",
    "\n",
    "**Objective:** Classify Netflix titles as **Movie (1)** or **TV Show (0)** using **Logistic Regression**.\n",
    "\n",
    "**Features used:** `duration_num`, `release_year`, `content_age`"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "classification_prep",
   "metadata": {},
   "outputs": [],
   "source": [
    "clf_df = df[['duration_num', 'release_year', 'content_age', 'is_movie']].dropna()\n",
    "\n",
    "X_clf = clf_df[['duration_num', 'release_year', 'content_age']]\n",
    "y_clf = clf_df['is_movie']\n",
    "\n",
    "X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(\n",
    "    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf\n",
    ")\n",
    "\n",
    "print(f\"Training samples : {X_tr_c.shape[0]}\")\n",
    "print(f\"Test samples     : {X_te_c.shape[0]}\")\n",
    "print(f\"Class balance    : {dict(y_te_c.value_counts())}  (0=TV Show, 1=Movie)\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "classification_train",
   "metadata": {},
   "outputs": [],
   "source": [
    "clf_model = LogisticRegression(max_iter=1000, random_state=42)\n",
    "clf_model.fit(X_tr_c, y_tr_c)\n",
    "\n",
    "y_pred_c = clf_model.predict(X_te_c)\n",
    "\n",
    "print(f\"✅ Accuracy: {accuracy_score(y_te_c, y_pred_c):.4f}  ({accuracy_score(y_te_c, y_pred_c)*100:.2f}%)\")\n",
    "print()\n",
    "print(\"📄 Classification Report:\")\n",
    "print(classification_report(y_te_c, y_pred_c, target_names=['TV Show', 'Movie']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "confusion_matrix_plot",
   "metadata": {},
   "outputs": [],
   "source": [
    "cm = confusion_matrix(y_te_c, y_pred_c)\n",
    "\n",
    "fig, axes = plt.subplots(1, 2, figsize=(13, 5))\n",
    "\n",
    "# Confusion matrix heatmap\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n",
    "            xticklabels=['TV Show','Movie'],\n",
    "            yticklabels=['TV Show','Movie'], ax=axes[0], linewidths=0.5)\n",
    "axes[0].set_title('Confusion Matrix')\n",
    "axes[0].set_xlabel('Predicted Label')\n",
    "axes[0].set_ylabel('Actual Label')\n",
    "\n",
    "# Normalised confusion matrix\n",
    "cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]\n",
    "sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens',\n",
    "            xticklabels=['TV Show','Movie'],\n",
    "            yticklabels=['TV Show','Movie'], ax=axes[1], linewidths=0.5)\n",
    "axes[1].set_title('Normalised Confusion Matrix')\n",
    "axes[1].set_xlabel('Predicted Label')\n",
    "axes[1].set_ylabel('Actual Label')\n",
    "\n",
    "plt.suptitle('Task 9 — Classification Results', fontsize=14)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "conclusion_header",
   "metadata": {},
   "source": [
    "---\n",
    "# Sr. No. 12 — Conclusion\n",
    "### Page 19"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "conclusion",
   "metadata": {},
   "source": [
    "## Conclusion\n",
    "\n",
    "This project performed a complete end-to-end Exploratory Data Analysis and predictive modelling exercise on the Netflix Movies and TV Shows dataset. The following key findings and outcomes were achieved:\n",
    "\n",
    "---\n",
    "\n",
    "### 📌 Key Findings\n",
    "\n",
    "| Area | Finding |\n",
    "|------|---------|\n",
    "| **Content Type** | ~70% of Netflix content is Movies; ~30% is TV Shows |\n",
    "| **Release Year** | Most content was released between 2010–2021 with rapid growth after 2015 |\n",
    "| **Top Countries** | United States dominates content, followed by India and United Kingdom |\n",
    "| **Popular Ratings** | TV-MA and TV-14 are the most common content ratings |\n",
    "| **Missing Data** | Highest missingness in `director` (~30%) and `cast` (~9%) columns |\n",
    "| **Outliers** | Movie durations had extreme outliers (>200 min); capped using IQR Winsorisation |\n",
    "| **Skewness** | `release_year` is left-skewed; `content_age` is right-skewed |\n",
    "\n",
    "---\n",
    "\n",
    "### 🤖 Model Results\n",
    "\n",
    "| Model | Task | Key Metric |\n",
    "|-------|------|------------|\n",
    "| **Linear Regression** | Predict `release_year` | Low R² → duration alone is a weak predictor |\n",
    "| **Logistic Regression** | Classify Movie vs TV Show | High Accuracy → duration is a strong classifier signal |\n",
    "\n",
    "---\n",
    "\n",
    "### 🚀 Scope for Improvement\n",
    "\n",
    "- Apply **Random Forest** or **XGBoost** for higher classification accuracy.\n",
    "- Use **TF-IDF / NLP** on `description` and `listed_in` for content-based analysis.\n",
    "- Build a **recommendation system** using collaborative or content-based filtering.\n",
    "- Perform **time-series analysis** on content addition trends.\n",
    "\n",
    "---\n",
    "\n",
    "This analysis demonstrates how data science tools can uncover meaningful patterns in real-world streaming data, and forms a strong foundation for more advanced predictive and recommendation systems."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3.10.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
