import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
import shap
from itertools import combinations
import time
import sys
sys.path.insert(0, '../')
from utils import mva_utils
from stxs_mva.data_processing import load_stxs_training_df, FEATURE_MAPPING

class ThreeClassClassifier:
    """3-class XGBoost classifier for VBF vs ggF vs Background classification."""
    
    def __init__(self):
        self.model = None
        self.feature_mapping = FEATURE_MAPPING
        self.feature_cols = list(self.feature_mapping.values())
        self.class_names = {0: 'Background', 1: 'VBF_H', 2: 'ggF_H'}
        
    def load_and_prepare_data(self):
        """Load DataFrame from pre-built Run 2 pickles (same flow as notebooks)."""
        # Load preprocessed training DataFrame (built beforehand)
        preprocessed_path = '/pscratch/sd/a/agarabag/ditdau_samples/stxs_training_run2.pkl'
        df = load_stxs_training_df(preprocessed_path)

        # Class reweighting
        if not df.empty:
            bg_count = len(df[df['label'] == 0]) * 2
            vbf_count = max(len(df[df['label'] == 1]), 1)
            ggf_count = max(len(df[df['label'] == 2]), 1)
            vbf_weight = bg_count / vbf_count
            ggf_weight = bg_count / ggf_count
            print(f"Class weights - VBF: {vbf_weight:.2f}, ggF: {ggf_weight:.2f}")
            df.loc[df['label'] == 1, 'combined_weights'] *= vbf_weight
            df.loc[df['label'] == 2, 'combined_weights'] *= ggf_weight

        # Create model feature columns (float32 for GPU efficiency)
        for human_name, feat_name in self.feature_mapping.items():
            if human_name in df.columns:
                df[feat_name] = df[human_name].astype('float32')
            else:
                df[feat_name] = np.nan
        # Drop events with negative or zero combined weights
        before = len(df)
        df = df[df['combined_weights'] > 0].reset_index(drop=True)
        print(f"Filtered non-positive weight events: {before - len(df)} removed, {len(df)} remain")
        return df
    
    def split_data_k_fold(self, df, k=3):
        """Simple k-fold split based on event number."""
        splits = []
        for i in range(k):
            fold_data = df[df['event_number'] % k == i].copy()
            splits.append(fold_data)
            
        print(f"Data split into {k} folds:")
        for i, fold in enumerate(splits):
            bg = len(fold[fold['label'] == 0])
            vbf = len(fold[fold['label'] == 1])
            ggf = len(fold[fold['label'] == 2])
            print(f"  Fold {i}: BG={bg}, VBF={vbf}, ggF={ggf}")
        return splits
    
    def train_model(self, X_train, y_train, sample_weights):
        """Train XGBoost classifier."""
        params = {
            'learning_rate': 0.05,
            'max_depth': 6,
            'n_estimators': 200,
            'objective': 'multi:softprob',
            'num_class': 3,
            'eval_metric': ['merror', 'mlogloss'],
            'random_state': 42,
            'gamma': 0.001,
            'verbosity': 1
        }
        
        self.model = XGBClassifier(**params)
        self.model.fit(X_train, y_train, sample_weight=sample_weights)
        return self.model
    
    def store_model(self, output_name):
        """Save trained model."""
        booster = self.model.get_booster()
        booster.dump_model(output_name)
        
    def generate_predictions(self, X_test):
        """Generate class predictions and probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        return y_pred, y_pred_proba
    
    def plot_confusion_matrix(self, y_true, y_pred, normalize=False, fold_idx=None):
        """Create confusion matrix visualization."""
        cm = confusion_matrix(y_true, y_pred, normalize='true' if normalize else None)
        
        plt.figure(figsize=(8, 6))
        labels = [self.class_names[i] for i in sorted(self.class_names.keys())]
        
        sns.heatmap(cm, annot=True, fmt='.3f' if normalize else 'd', 
                   cmap='Blues', xticklabels=labels, yticklabels=labels)
        
        plt.ylabel('True Class', fontsize=12)
        plt.xlabel('Predicted Class', fontsize=12)
        plt.title(f'Confusion Matrix {"(Normalized)" if normalize else "(Raw)"}')
        
        suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
        plt.savefig(f'confusion_matrix{suffix}_{"normalized" if normalize else "raw"}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_class_score_distributions(self, y_true, y_pred_proba, class_idx, class_name, fold_idx=None):
        """Plot BDT score distributions for each true class."""
        plt.figure(figsize=(10, 6))
        colors = ['blue', 'red', 'green']
        
        for true_class in [0, 1, 2]:
            mask = y_true == true_class
            scores = y_pred_proba[mask, class_idx]
            
            if len(scores) == 0:
                print(f"Warning: No samples found for true class {true_class}")
                continue
                
            plt.hist(scores, bins=30, alpha=0.7, color=colors[true_class],
                    label=f'True {self.class_names[true_class]} (n={len(scores)})', 
                    histtype='step', density=True, linewidth=2)
        
        plt.xlabel(f'BDT Score for {class_name} Class', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.title(f'Score Distribution: {class_name} Classifier Output')
        
        suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
        plt.savefig(f'score_distribution_{class_name.lower()}{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curves(self, y_true, y_pred_proba, fold_idx=None):
        """Plot ROC curves for all class pairs."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        class_pairs = list(combinations([0, 1, 2], 2))
        colors = ['blue', 'red', 'green']
        
        for i, (class1, class2) in enumerate(class_pairs):
            mask = (y_true == class1) | (y_true == class2)
            if mask.sum() == 0:
                continue
                
            y_binary = (y_true[mask] == class1).astype(int)
            scores = y_pred_proba[mask, class1]
            fpr, tpr, _ = roc_curve(y_binary, scores)
            roc_auc = auc(fpr, tpr)
            
            class1_name = self.class_names[class1]
            class2_name = self.class_names[class2]
            
            axes[0].plot(fpr, tpr, linewidth=2, color=colors[i],
                       label=f"{class1_name} vs {class2_name} (AUC={roc_auc:.3f})")
            
            # Background rejection
            valid_mask = fpr > 1e-6
            if valid_mask.sum() > 0:
                rejection = 1 / fpr[valid_mask]
                axes[1].plot(tpr[valid_mask], rejection, linewidth=2, color=colors[i],
                           label=f"{class1_name} vs {class2_name} (AUC={roc_auc:.3f})")
        
        axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[0].set_xlabel('False Positive Rate', fontsize=12)
        axes[0].set_ylabel('True Positive Rate', fontsize=12)
        axes[0].set_title('ROC Curves', fontsize=14)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        axes[1].set_xlabel('Signal Efficiency', fontsize=12)
        axes[1].set_ylabel('Background Rejection', fontsize=12)
        axes[1].set_yscale('log')
        axes[1].set_title('Background Rejection Curves', fontsize=14)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
        plt.savefig(f'roc_curves{suffix}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_feature_importance(self, X_test, fold_idx=None):
        """Analyze feature importance using SHAP."""
        X_test_ordered = X_test[self.feature_cols]
        inv_map = {v: k for k, v in self.feature_mapping.items()}
        X_sample_named = X_test_ordered.rename(columns=inv_map)

        explainer = shap.Explainer(self.model)
        shap_values = explainer(X_test_ordered)

        for class_idx, class_name in enumerate(['Bkg', 'VBF', 'ggF']):
            shap.summary_plot(
                shap_values.values[:, :, class_idx],
                X_sample_named,
                feature_names=X_sample_named.columns,
                plot_type="bar",
                show=False
            )
            plt.title(f"SHAP Feature Importance for {class_name}")
            suffix = f"_fold{fold_idx}" if fold_idx is not None else ""
            plt.savefig(f'shap_summary_{class_name}{suffix}.png', dpi=300, bbox_inches='tight')
            plt.close()


def main():
    """Main execution pipeline."""
    start_time = time.time()
    classifier = ThreeClassClassifier()
    
    # Load and prepare data
    print("Loading and preparing data...")
    df = classifier.load_and_prepare_data()
    
    # Split data
    data_splits = classifier.split_data_k_fold(df, k=3)
    
    for i in range(3): 
        print(f"\n========== Fold {i+1}/3 ==========")
        
        # Use fold i as test, others as training
        test_data = data_splits[i]
        train_data = pd.concat([data_splits[j] for j in range(3) if j != i])
        
        # Filter out non-positive combined weights in both train and test
        train_data = train_data[train_data['combined_weights'] > 0]
        test_data = test_data[test_data['combined_weights'] > 0]

        # Prepare data
        X_train = train_data[classifier.feature_cols].astype('float32')
        y_train = train_data['label']
        weights_train = (train_data['combined_weights'] * train_data['fake_factor']).astype(float)
        
        X_test = test_data[classifier.feature_cols].astype('float32')
        y_test = test_data['label']
                
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Class distribution in test set:")
        for class_id, count in pd.Series(y_test).value_counts().sort_index().items():
            print(f"  {classifier.class_names[class_id]}: {count}")
            
        # Train model
        print("Training XGBoost model...")
        classifier.train_model(X_train, y_train, weights_train)
        
        # Store model
        classifier.store_model(f'xgboost_3class_model_fold_{i}.txt')
        
        # Generate predictions
        y_pred, y_pred_proba = classifier.generate_predictions(X_test)
        
        print(f"Prediction probabilities shape: {y_pred_proba.shape}")
        print(f"Probability ranges:")
        for k in range(3):
            print(f"  Class {classifier.class_names[k]}: [{y_pred_proba[:, k].min():.3f}, {y_pred_proba[:, k].max():.3f}]")
                
        # Create visualizations
        print("Generating visualizations...")
        classifier.plot_confusion_matrix(y_test, y_pred, normalize=True, fold_idx=i)
        
        for class_idx, class_name in classifier.class_names.items():
            classifier.plot_class_score_distributions(y_test, y_pred_proba, class_idx, class_name, fold_idx=i)
        
        classifier.plot_roc_curves(y_test, y_pred_proba, fold_idx=i)
        classifier.analyze_feature_importance(X_test, fold_idx=i)
    
    print("Analysis complete! Check generated plots.")
    print(f"--- {time.time() - start_time:.2f} seconds ---")


if __name__ == "__main__":
    main()