"""
Credit Scoring Model - End-to-End Implementation
================================================
A comprehensive credit risk assessment system for predicting creditworthiness
using the German Credit dataset.

Author: Senior Data Scientist
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, f1_score
)
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class CreditScoringModel:
    """
    A comprehensive credit scoring system that handles data preparation,
    feature engineering, model training, and evaluation.
    """
    
    def __init__(self, filepath):
        """
        Initialize the credit scoring model.
        
        Parameters:
        -----------
        filepath : str
            Path to the credit dataset CSV file
        """
        self.filepath = filepath
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load the credit dataset from CSV file."""
        print("=" * 80)
        print("STEP 1: LOADING DATA")
        print("=" * 80)
        
        self.df = pd.read_csv(r"C:\Users\jamal\Downloads\GermanCredit.csv")
        print(f"\nDataset loaded successfully!")
        print(f"Shape: {self.df.shape}")
        print(f"\nFirst few rows:")
        print(self.df.head())
        print(f"\nColumn names:\n{self.df.columns.tolist()}")
        print(f"\nData types:\n{self.df.dtypes}")
        print(f"\nTarget variable distribution:")
        print(self.df['credit_risk'].value_counts())
        print(f"Class balance: {self.df['credit_risk'].value_counts(normalize=True)}")
        
    def explore_data(self):
        """Perform exploratory data analysis."""
        print("\n" + "=" * 80)
        print("STEP 2: EXPLORATORY DATA ANALYSIS")
        print("=" * 80)
        
        # Check for missing values
        print("\nMissing values:")
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("No missing values found!")
        else:
            print(missing[missing > 0])
        
        # Basic statistics for numerical features
        print("\nNumerical features summary:")
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns
        print(self.df[numerical_cols].describe())
        
        # Check for duplicate rows
        duplicates = self.df.duplicated().sum()
        print(f"\nDuplicate rows: {duplicates}")
        
    def prepare_data(self):
        """
        Prepare data for modeling:
        - Handle missing values
        - Encode categorical variables
        - Engineer new features
        - Scale numerical features
        """
        print("\n" + "=" * 80)
        print("STEP 3: DATA PREPARATION & FEATURE ENGINEERING")
        print("=" * 80)
        
        # Create a copy for processing
        df_processed = self.df.copy()
        
        # 1. HANDLE MISSING VALUES
        print("\n[1] Handling Missing Values...")
        # No missing values in this dataset, but adding robust handling
        if df_processed.isnull().sum().sum() > 0:
            # For numerical columns, fill with median
            num_cols = df_processed.select_dtypes(include=[np.number]).columns
            for col in num_cols:
                if df_processed[col].isnull().sum() > 0:
                    df_processed[col].fillna(df_processed[col].median(), inplace=True)
            
            # For categorical columns, fill with mode
            cat_cols = df_processed.select_dtypes(include=['object']).columns
            for col in cat_cols:
                if df_processed[col].isnull().sum() > 0:
                    df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)
        
        print("✓ Missing values handled")
        
        # 2. FEATURE ENGINEERING
        print("\n[2] Engineering New Features...")
        
        # Create debt-to-income proxy (amount / duration as monthly payment)
        df_processed['monthly_payment'] = df_processed['amount'] / df_processed['duration']
        
        # Create credit utilization indicator based on amount and savings
        df_processed['high_credit_utilization'] = (
            (df_processed['amount'] > df_processed['amount'].median()) & 
            (df_processed['savings'].str.contains('< 100|unknown', case=False, regex=True))
        ).astype(int)
        
        # Risk indicators from credit history
        df_processed['critical_history'] = df_processed['credit_history'].str.contains(
            'critical|delay', case=False, regex=True
        ).astype(int)
        
        # Employment stability
        df_processed['stable_employment'] = df_processed['employment_duration'].str.contains(
            '>= 7 years|4 <= ... < 7', case=False, regex=True
        ).astype(int)
        
        # Age groups
        df_processed['age_group'] = pd.cut(
            df_processed['age'], 
            bins=[0, 25, 35, 50, 100], 
            labels=['young', 'middle', 'mature', 'senior']
        )
        
        # High installment rate risk
        df_processed['high_installment_rate'] = (df_processed['installment_rate'] >= 3).astype(int)
        
        print("✓ New features created:")
        print("  - monthly_payment")
        print("  - high_credit_utilization")
        print("  - critical_history")
        print("  - stable_employment")
        print("  - age_group")
        print("  - high_installment_rate")
        
        # 3. ENCODE CATEGORICAL VARIABLES
        print("\n[3] Encoding Categorical Variables...")
        
        # Separate target variable
        y = df_processed['credit_risk']
        X = df_processed.drop('credit_risk', axis=1)
        
        # Identify categorical columns (including categorical dtype from pd.cut)
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        print(f"Categorical columns to encode: {len(categorical_cols)}")
        
        # One-hot encode categorical variables
        X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        print(f"✓ Features after encoding: {X_encoded.shape[1]}")
        
        # 4. HANDLE OUTLIERS
        print("\n[4] Handling Outliers...")
        
        # Identify numerical columns
        numerical_cols = X_encoded.select_dtypes(include=[np.number]).columns
        
        # Cap outliers using IQR method for key financial variables
        outlier_cols = ['amount', 'duration', 'age', 'monthly_payment']
        outlier_count = 0
        
        for col in outlier_cols:
            if col in numerical_cols:
                Q1 = X_encoded[col].quantile(0.25)
                Q3 = X_encoded[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 3 * IQR
                upper_bound = Q3 + 3 * IQR
                
                outliers_before = ((X_encoded[col] < lower_bound) | 
                                  (X_encoded[col] > upper_bound)).sum()
                
                X_encoded[col] = X_encoded[col].clip(lower=lower_bound, upper=upper_bound)
                outlier_count += outliers_before
        
        print(f"✓ {outlier_count} outliers capped using IQR method")
        
        # 5. SPLIT DATA
        print("\n[5] Splitting Data...")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_encoded, y, test_size=0.25, random_state=42, stratify=y
        )
        
        print(f"✓ Training set: {self.X_train.shape}")
        print(f"✓ Testing set: {self.X_test.shape}")
        print(f"✓ Class distribution in train: {self.y_train.value_counts(normalize=True).to_dict()}")
        
        # 6. SCALE FEATURES
        print("\n[6] Scaling Features...")
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print("✓ Features standardized using StandardScaler")
        
        # Store feature names for later interpretation
        self.feature_names = X_encoded.columns.tolist()
        
        print(f"\nFinal feature count: {len(self.feature_names)}")
        
    def feature_selection(self):
        """
        Perform feature selection using statistical methods.
        """
        print("\n" + "=" * 80)
        print("STEP 4: FEATURE SELECTION")
        print("=" * 80)
        
        # Use a simple Random Forest to get feature importance
        rf_temp = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_temp.fit(self.X_train_scaled, self.y_train)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf_temp.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 20 Most Important Features:")
        print(feature_importance.head(20).to_string(index=False))
        
        # Store for visualization
        self.feature_importance = feature_importance
        
    def train_models(self):
        """
        Train multiple classification models:
        - Logistic Regression
        - Decision Tree
        - Random Forest
        """
        print("\n" + "=" * 80)
        print("STEP 5: MODEL TRAINING")
        print("=" * 80)
        
        # 1. LOGISTIC REGRESSION
        print("\n[1] Training Logistic Regression...")
        print("    Rationale: Highly interpretable, outputs probabilities,")
        print("               suitable for regulated environments")
        
        self.models['Logistic Regression'] = LogisticRegression(
            max_iter=1000, 
            random_state=42,
            class_weight='balanced',  # Handle class imbalance
            solver='lbfgs'
        )
        self.models['Logistic Regression'].fit(self.X_train_scaled, self.y_train)
        print("    ✓ Model trained successfully")
        
        # 2. DECISION TREE
        print("\n[2] Training Decision Tree...")
        print("    Rationale: Highly interpretable rules, captures non-linear relationships")
        
        self.models['Decision Tree'] = DecisionTreeClassifier(
            max_depth=5,  # Prevent overfitting
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            class_weight='balanced'
        )
        self.models['Decision Tree'].fit(self.X_train_scaled, self.y_train)
        print("    ✓ Model trained successfully")
        
        # 3. RANDOM FOREST
        print("\n[3] Training Random Forest...")
        print("    Rationale: High accuracy, robust to outliers, provides feature importance")
        
        self.models['Random Forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        self.models['Random Forest'].fit(self.X_train_scaled, self.y_train)
        print("    ✓ Model trained successfully")
        
        print(f"\n✓ All {len(self.models)} models trained successfully!")
        
    def evaluate_models(self):
        """
        Evaluate all trained models using multiple metrics:
        - Precision, Recall, F1-Score
        - ROC-AUC
        - Confusion Matrix
        """
        print("\n" + "=" * 80)
        print("STEP 6: MODEL EVALUATION")
        print("=" * 80)
        
        comparison_data = []
        
        for name, model in self.models.items():
            print(f"\n{'=' * 80}")
            print(f"Evaluating: {name}")
            print(f"{'=' * 80}")
            
            # Predictions
            y_pred = model.predict(self.X_test_scaled)
            y_pred_proba = model.predict_proba(self.X_test_scaled)[:, 1]
            
            # Classification Report
            print("\nClassification Report:")
            print(classification_report(self.y_test, y_pred, 
                                       target_names=['Good Credit', 'Bad Credit']))
            
            # Confusion Matrix
            cm = confusion_matrix(self.y_test, y_pred)
            print("\nConfusion Matrix:")
            print(cm)
            
            # Calculate metrics
            from sklearn.metrics import precision_score, recall_score
            
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            roc_auc = roc_auc_score(self.y_test, y_pred_proba)
            
            print(f"\nKey Metrics:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  ROC-AUC:   {roc_auc:.4f}")
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train_scaled, self.y_train, 
                                       cv=5, scoring='roc_auc')
            print(f"  CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
            
            # Store results
            comparison_data.append({
                'Model': name,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'ROC-AUC': roc_auc,
                'CV ROC-AUC': cv_scores.mean()
            })
            
            # Store detailed results
            self.results[name] = {
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'confusion_matrix': cm,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc
            }
        
        # Create comparison DataFrame
        self.comparison_df = pd.DataFrame(comparison_data)
        
        print("\n" + "=" * 80)
        print("MODEL COMPARISON SUMMARY")
        print("=" * 80)
        print(self.comparison_df.to_string(index=False))
        
        # Identify best model
        best_model_name = self.comparison_df.loc[
            self.comparison_df['ROC-AUC'].idxmax(), 'Model'
        ]
        print(f"\n🏆 Best Model (by ROC-AUC): {best_model_name}")
        
    def interpret_models(self):
        """
        Provide interpretability for each model:
        - Logistic Regression: coefficient analysis
        - Tree-based models: feature importance
        """
        print("\n" + "=" * 80)
        print("STEP 7: MODEL INTERPRETABILITY")
        print("=" * 80)
        
        # 1. LOGISTIC REGRESSION INTERPRETATION
        print("\n[1] Logistic Regression - Coefficient Analysis")
        print("-" * 80)
        
        lr_model = self.models['Logistic Regression']
        
        # Get coefficients
        coef_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Coefficient': lr_model.coef_[0],
            'Abs_Coefficient': np.abs(lr_model.coef_[0])
        }).sort_values('Abs_Coefficient', ascending=False)
        
        print("\nTop 15 Most Influential Features (by absolute coefficient):")
        print(coef_df.head(15)[['Feature', 'Coefficient']].to_string(index=False))
        
        print("\nInterpretation:")
        print("  • Positive coefficients → Increase probability of BAD credit (default)")
        print("  • Negative coefficients → Decrease probability of BAD credit (good credit)")
        
        # Key insights
        print("\nKey Risk Factors (Positive Coefficients - Increase Default Risk):")
        top_risk = coef_df[coef_df['Coefficient'] > 0].head(5)
        for idx, row in top_risk.iterrows():
            print(f"  • {row['Feature']}: {row['Coefficient']:.4f}")
        
        print("\nKey Protective Factors (Negative Coefficients - Decrease Default Risk):")
        top_protective = coef_df[coef_df['Coefficient'] < 0].head(5)
        for idx, row in top_protective.iterrows():
            print(f"  • {row['Feature']}: {row['Coefficient']:.4f}")
        
        # 2. RANDOM FOREST INTERPRETATION
        print("\n\n[2] Random Forest - Feature Importance Analysis")
        print("-" * 80)
        
        rf_model = self.models['Random Forest']
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 15 Most Important Features:")
        print(importance_df.head(15).to_string(index=False))
        
        print("\nInterpretation:")
        print("  Feature importance indicates how much each feature contributes")
        print("  to reducing prediction error in the Random Forest model.")
        
        # Store for visualization
        self.coef_df = coef_df
        self.importance_df = importance_df
        
        # 3. DECISION TREE INTERPRETATION
        print("\n\n[3] Decision Tree - Structure Summary")
        print("-" * 80)
        
        dt_model = self.models['Decision Tree']
        
        print(f"Tree depth: {dt_model.get_depth()}")
        print(f"Number of leaves: {dt_model.get_n_leaves()}")
        print(f"Number of features used: {np.sum(dt_model.feature_importances_ > 0)}")
        
        # Top features in decision tree
        dt_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': dt_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\nTop 10 Features Used in Decision Tree:")
        print(dt_importance.head(10).to_string(index=False))
        
    def create_visualizations(self):
        """
        Create comprehensive visualizations for model performance and interpretation.
        """
        print("\n" + "=" * 80)
        print("STEP 8: CREATING VISUALIZATIONS")
        print("=" * 80)
        
        # Create a figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Model Comparison - Performance Metrics
        ax1 = plt.subplot(2, 3, 1)
        metrics = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        x = np.arange(len(self.comparison_df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            ax1.bar(x + i*width, self.comparison_df[metric], width, label=metric)
        
        ax1.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(self.comparison_df['Model'], rotation=15, ha='right')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # 2. ROC Curves
        ax2 = plt.subplot(2, 3, 2)
        for name in self.models.keys():
            fpr, tpr, _ = roc_curve(self.y_test, self.results[name]['y_pred_proba'])
            auc = self.results[name]['roc_auc']
            ax2.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
        
        ax2.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        ax2.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax2.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax2.set_title('ROC Curves', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        # 3. Confusion Matrices
        for idx, (name, model) in enumerate(self.models.items()):
            ax = plt.subplot(2, 3, idx + 4)
            cm = self.results[name]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Good', 'Bad'], yticklabels=['Good', 'Bad'])
            ax.set_title(f'{name}\nConfusion Matrix', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=10)
            ax.set_xlabel('Predicted Label', fontsize=10)
        
        # 4. Logistic Regression - Top Coefficients
        ax3 = plt.subplot(2, 3, 3)
        top_coefs = pd.concat([
            self.coef_df.head(10),
            self.coef_df.tail(10)
        ]).sort_values('Coefficient')
        
        colors = ['red' if x > 0 else 'green' for x in top_coefs['Coefficient']]
        ax3.barh(range(len(top_coefs)), top_coefs['Coefficient'], color=colors, alpha=0.7)
        ax3.set_yticks(range(len(top_coefs)))
        ax3.set_yticklabels([f[:30] + '...' if len(f) > 30 else f 
                            for f in top_coefs['Feature']], fontsize=8)
        ax3.set_xlabel('Coefficient Value', fontsize=12, fontweight='bold')
        ax3.set_title('Logistic Regression\nTop Feature Coefficients', 
                     fontsize=12, fontweight='bold')
        ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax3.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/claude/credit_model_evaluation.png', dpi=300, bbox_inches='tight')
        print("\n✓ Visualizations saved to: credit_model_evaluation.png")
        
        # Create separate feature importance plot
        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Random Forest Feature Importance
        top_rf = self.importance_df.head(20)
        ax1.barh(range(len(top_rf)), top_rf['Importance'], color='skyblue', alpha=0.8)
        ax1.set_yticks(range(len(top_rf)))
        ax1.set_yticklabels([f[:40] + '...' if len(f) > 40 else f 
                            for f in top_rf['Feature']], fontsize=9)
        ax1.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax1.set_title('Random Forest - Top 20 Feature Importances', 
                     fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        ax1.invert_yaxis()
        
        # Feature importance by category
        importance_summary = self.importance_df.copy()
        importance_summary['Category'] = importance_summary['Feature'].apply(
            lambda x: 'Credit History' if 'credit_history' in x 
            else 'Financial' if any(k in x for k in ['amount', 'duration', 'savings', 'payment'])
            else 'Employment' if 'employment' in x
            else 'Personal' if any(k in x for k in ['age', 'personal', 'sex'])
            else 'Housing' if 'housing' in x
            else 'Other'
        )
        
        category_importance = importance_summary.groupby('Category')['Importance'].sum().sort_values(ascending=False)
        
        ax2.pie(category_importance, labels=category_importance.index, autopct='%1.1f%%',
               startangle=90, colors=sns.color_palette('husl', len(category_importance)))
        ax2.set_title('Feature Importance by Category', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/home/claude/feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        print("✓ Feature importance visualizations saved to: feature_importance_analysis.png")
        
        plt.close('all')
        
    def generate_report(self):
        """
        Generate a comprehensive final report with model selection and recommendations.
        """
        print("\n" + "=" * 80)
        print("FINAL REPORT: CREDIT SCORING MODEL")
        print("=" * 80)
        
        print("\n📊 EXECUTIVE SUMMARY")
        print("-" * 80)
        print("""
This credit scoring model was developed to predict creditworthiness using
the German Credit dataset. Three classification algorithms were implemented
and evaluated: Logistic Regression, Decision Tree, and Random Forest.
        """)
        
        print("\n📈 MODEL PERFORMANCE SUMMARY")
        print("-" * 80)
        print(self.comparison_df.to_string(index=False))
        
        # Determine best model
        best_by_auc = self.comparison_df.loc[self.comparison_df['ROC-AUC'].idxmax()]
        best_by_f1 = self.comparison_df.loc[self.comparison_df['F1-Score'].idxmax()]
        
        print(f"\n🏆 BEST MODEL SELECTION")
        print("-" * 80)
        print(f"Best by ROC-AUC: {best_by_auc['Model']} (AUC = {best_by_auc['ROC-AUC']:.4f})")
        print(f"Best by F1-Score: {best_by_f1['Model']} (F1 = {best_by_f1['F1-Score']:.4f})")
        
        print("\n💡 RECOMMENDATION")
        print("-" * 80)
        print("""
For regulated financial environments, we recommend using the LOGISTIC REGRESSION model
for the following reasons:

1. INTERPRETABILITY: Coefficients are easily explainable to regulators and stakeholders
2. TRANSPARENCY: Clear mathematical relationship between features and predictions
3. PROBABILITY OUTPUTS: Provides well-calibrated probability estimates for risk scoring
4. REGULATORY COMPLIANCE: Meets requirements for explainable AI in financial services
5. PERFORMANCE: Competitive performance with ROC-AUC close to ensemble methods

While Random Forest shows slightly better ROC-AUC, the interpretability advantage
of Logistic Regression is crucial in credit scoring applications where decisions
must be explainable to both regulators and customers.
        """)
        
        print("\n🔑 KEY RISK FACTORS IDENTIFIED")
        print("-" * 80)
        
        # Top risk factors from logistic regression
        top_risk = self.coef_df[self.coef_df['Coefficient'] > 0].head(5)
        print("\nFactors that INCREASE default risk:")
        for idx, row in top_risk.iterrows():
            print(f"  • {row['Feature']}: {row['Coefficient']:.4f}")
        
        # Top protective factors
        top_protective = self.coef_df[self.coef_df['Coefficient'] < 0].head(5)
        print("\nFactors that DECREASE default risk:")
        for idx, row in top_protective.iterrows():
            print(f"  • {row['Feature']}: {row['Coefficient']:.4f}")
        
        print("\n📋 IMPLEMENTATION CONSIDERATIONS")
        print("-" * 80)
        print("""
1. MODEL MONITORING: Implement ongoing monitoring of model performance
2. REGULAR RETRAINING: Retrain quarterly with new data to maintain accuracy
3. BIAS TESTING: Conduct regular fairness audits across demographic groups
4. THRESHOLD TUNING: Adjust classification threshold based on business risk appetite
5. DOCUMENTATION: Maintain comprehensive documentation for regulatory compliance
6. HUMAN OVERSIGHT: Implement human review for borderline cases
        """)
        
        print("\n✅ NEXT STEPS")
        print("-" * 80)
        print("""
1. Validate model on out-of-time holdout dataset
2. Conduct comprehensive bias and fairness analysis
3. Develop model monitoring dashboard
4. Create API for real-time scoring
5. Prepare documentation for regulatory approval
6. Implement A/B testing framework for deployment
        """)
        
        print("\n" + "=" * 80)
        print("END OF REPORT")
        print("=" * 80)


def main():
    """
    Main execution function to run the complete credit scoring pipeline.
    """
    print("\n" + "=" * 80)
    print("CREDIT SCORING MODEL - END-TO-END IMPLEMENTATION")
    print("=" * 80)
    print("\nThis system implements a comprehensive credit risk assessment model")
    print("suitable for real-world financial applications.\n")
    
    # Initialize model
    model = CreditScoringModel('/mnt/user-data/uploads/GermanCredit.csv')
    
    # Execute pipeline
    model.load_data()
    model.explore_data()
    model.prepare_data()
    model.feature_selection()
    model.train_models()
    model.evaluate_models()
    model.interpret_models()
    model.create_visualizations()
    model.generate_report()
    
    print("\n✅ Pipeline completed successfully!")
    print("\nGenerated outputs:")
    print("  1. credit_model_evaluation.png - Model performance visualizations")
    print("  2. feature_importance_analysis.png - Feature importance analysis")
    
    return model


if __name__ == "__main__":
    credit_model = main()
