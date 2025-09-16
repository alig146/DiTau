import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.preprocessing import label_binarize
from xgboost import XGBClassifier
import shap
from itertools import combinations
import awkward as ak
import time
import sys
sys.path.insert(0, '/global/homes/a/agarabag/DiTau')
from utils import mva_utils

class ThreeClassClassifier:
    """3-class XGBoost classifier for VBF vs ggF vs Background classification."""
    
    def __init__(self):
        self.model = None
        self.feature_mapping = {
            'leadsubjet_pt': 'f0',
            'subleadsubjet_pt': 'f1', 
            'visible_ditau_m': 'f2',
            'collinear_mass': 'f3',
            'delta_R': 'f4',
            'met': 'f5',
            'delta_phi_met_ditau': 'f6',
            'eta_product': 'f7',
            'delta_eta_jj': 'f8',
            'Mjj': 'f9',
            'pt_jj': 'f10',
            'pt_jj_higgs': 'f11'
        }
        self.feature_cols = list(self.feature_mapping.values())
        self.class_names = {0: 'Background', 1: 'VBF_H', 2: 'ggF_H'}
        
    def _arrays_to_df(self, arr_list, label):
        """Convert mva_utils.Var output to DataFrame."""
        var_names = [
            'ditau_pt','leadsubjet_pt','subleadsubjet_pt','visible_ditau_m','met','collinear_mass','x1','x2',
            'met_sig','met_phi','event_number','k_t','kappa','delta_R','delta_phi','delta_eta','combined_weights','fake_factor',
            'delta_R_lead','delta_eta_lead','delta_phi_lead','delta_R_sublead','delta_eta_sublead','delta_phi_sublead',
            'met_centrality','omni_score','leadsubjet_charge','subleadsubjet_charge','leadsubjet_n_core_tracks','subleadsubjet_n_core_tracks',
            'e_ratio_lead','e_ratio_sublead','higgs_pt','leadsubjet_eta','subleadsubjet_eta','ditau_eta','delta_phi_met_ditau',
            'Ht','eta_product','delta_eta_jj','Mjj','pt_jj','delta_phi_jj','pt_jj_higgs','delta_r_leadjet_ditau','leading_jet_pt','subleading_jet_pt'
        ]
        
        data = {}
        for name, arr in zip(var_names, arr_list):
            try:
                if hasattr(arr, 'to_numpy'):
                    data[name] = arr.to_numpy()
                elif isinstance(arr, ak.Array):
                    data[name] = ak.to_numpy(arr)
                else:
                    data[name] = np.array(arr)
            except Exception:
                data[name] = np.array(arr)
        
        df = pd.DataFrame(data)
        df['label'] = label
        return df

    def load_and_prepare_data(self):
        """Load Run 2 samples with the same flow as the notebook: apply_cuts -> combine -> Var/Data_Var."""
        include_vbf = True
        cfg_run2 = mva_utils.get_config('run2')

        # Define DSID groups (Run 2)
        ggH = mva_utils.ggH
        VBFH = mva_utils.VBFH
        background_groups = {
            'Ztt_inc': mva_utils.Ztt_inc,
            'ttV': mva_utils.ttV,
            'VV': mva_utils.VV,
            'Top': mva_utils.Top,
            'W': mva_utils.W,
            'Zll_inc': mva_utils.Zll_inc,
        }

        # Datasets and year suffixes
        datasets = [('mc20a','15','a'), ('mc20d','17','d'), ('mc20e','18','e')]

        # 1) Build uncut MC payload per dataset key
        uncut_mc = {ds_key: {} for ds_key, _, _ in datasets}
        for ds_key, year, suffix in datasets:
            # Weights per DSID for this year
            sig_ws = {**mva_utils.fetch_weights(ggH, year), **mva_utils.fetch_weights(VBFH, year)}
            if ggH:
                uncut_mc[ds_key][f'ggH_{suffix}'] = mva_utils.read_root(ggH, sig_ws, year_id=ds_key, year=year, is_signal=True, config=cfg_run2)
            if VBFH:
                uncut_mc[ds_key][f'VBFH_{suffix}'] = mva_utils.read_root(VBFH, sig_ws, year_id=ds_key, year=year, is_signal=True, config=cfg_run2)
            for name, dsids in background_groups.items():
                if dsids:
                    ws = mva_utils.fetch_weights(dsids, year)
                    uncut_mc[ds_key][f'{name}_{suffix}'] = mva_utils.read_root(dsids, ws, year_id=ds_key, year=year, is_signal=False, config=cfg_run2)

        # 2) Apply cuts and combine across years
        cut_mc = mva_utils.apply_cuts(uncut_mc, data_type='MC', config=cfg_run2)
        combined_mc = mva_utils.combine_mc_years(cut_mc)

        # 3) Data path mirrors the notebook
        uncut_data = {
            'data_15': mva_utils.read_data_root(year='15', config=cfg_run2),
            'data_16': mva_utils.read_data_root(year='16', config=cfg_run2),
            'data_17': mva_utils.read_data_root(year='17', config=cfg_run2),
            'data_18': mva_utils.read_data_root(year='18', config=cfg_run2),
        }
        cut_data = mva_utils.apply_cuts(uncut_data, data_type='data', config=cfg_run2)
        combined_data = mva_utils.combine_data_years(cut_data)

        # 4) Assemble DataFrame from variables
        dfs = []
        if 'ggH' in combined_mc:
            var_ggh = mva_utils.Var(combined_mc['ggH'], include_vbf=include_vbf)
            dfs.append(self._arrays_to_df(var_ggh, label=2))
        if 'VBFH' in combined_mc:
            var_vbf = mva_utils.Var(combined_mc['VBFH'], include_vbf=include_vbf)
            dfs.append(self._arrays_to_df(var_vbf, label=1))
        for key, arr in combined_mc.items():
            if key in ['ggH', 'VBFH']:
                continue
            var_bkg = mva_utils.Var(arr, include_vbf=include_vbf)
            dfs.append(self._arrays_to_df(var_bkg, label=0))
        if len(combined_data) > 0:
            var_data = mva_utils.Data_Var(combined_data, include_vbf=include_vbf)
            dfs.append(self._arrays_to_df(var_data, label=0))

        df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        # 5) Class reweighting
        if not df.empty:
            bg_count = len(df[df['label'] == 0]) * 2
            vbf_count = max(len(df[df['label'] == 1]), 1)
            ggf_count = max(len(df[df['label'] == 2]), 1)
            vbf_weight = bg_count / vbf_count
            ggf_weight = bg_count / ggf_count
            print(f"Class weights - VBF: {vbf_weight:.2f}, ggF: {ggf_weight:.2f}")
            df.loc[df['label'] == 1, 'combined_weights'] *= vbf_weight
            df.loc[df['label'] == 2, 'combined_weights'] *= ggf_weight

        # 6) Create model feature columns
        for human_name, feat_name in self.feature_mapping.items():
            if human_name in df.columns:
                df[feat_name] = df[human_name]
            else:
                df[feat_name] = np.nan
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
        
        # Prepare data
        X_train = train_data[classifier.feature_cols]
        y_train = train_data['label']
        weights_train = train_data['combined_weights'] * train_data['fake_factor']
        
        X_test = test_data[classifier.feature_cols]
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