"""
Meta-Analysis Implementation Following Cochrane Handbook Guidelines
Author: Meta-Analysis Tool
Date: 2024

This implementation follows the Cochrane Handbook for Systematic Reviews of Interventions
and provides comprehensive meta-analysis capabilities including:
- Effect size calculations (OR, RR, MD, SMD)
- Heterogeneity assessment (I², Q-test, τ²)
- Fixed and random effects models
- Forest plot visualization
- Sensitivity analysis
- Publication bias assessment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2
import warnings
warnings.filterwarnings('ignore')

class MetaAnalysis:
    """
    Comprehensive meta-analysis class following Cochrane guidelines
    """
    
    def __init__(self, data=None):
        """
        Initialize meta-analysis
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with columns: study, n1, n2, events1, events2 (for binary outcomes)
            or study, n1, n2, mean1, mean2, sd1, sd2 (for continuous outcomes)
        """
        self.data = data
        self.effect_sizes = None
        self.weights = None
        self.heterogeneity_stats = {}
        self.fixed_effect = {}
        self.random_effect = {}
        self.forest_plot_data = None
        
    def load_binary_data(self, study_names, n1, n2, events1, events2):
        """
        Load binary outcome data (e.g., mortality, response rates)
        
        Parameters:
        -----------
        study_names : list
            Names of studies
        n1, n2 : list
            Sample sizes for treatment and control groups
        events1, events2 : list
            Number of events in treatment and control groups
        """
        self.data = pd.DataFrame({
            'study': study_names,
            'n1': n1,
            'n2': n2,
            'events1': events1,
            'events2': events2
        })
        self.outcome_type = 'binary'
        
    def load_continuous_data(self, study_names, n1, n2, mean1, mean2, sd1, sd2):
        """
        Load continuous outcome data (e.g., blood pressure, pain scores)
        
        Parameters:
        -----------
        study_names : list
            Names of studies
        n1, n2 : list
            Sample sizes for treatment and control groups
        mean1, mean2 : list
            Means for treatment and control groups
        sd1, sd2 : list
            Standard deviations for treatment and control groups
        """
        self.data = pd.DataFrame({
            'study': study_names,
            'n1': n1,
            'n2': n2,
            'mean1': mean1,
            'mean2': mean2,
            'sd1': sd1,
            'sd2': sd2
        })
        self.outcome_type = 'continuous'
    
    def calculate_effect_sizes(self, effect_measure='OR'):
        """
        Calculate effect sizes and their standard errors
        
        Parameters:
        -----------
        effect_measure : str
            'OR' for odds ratio, 'RR' for risk ratio, 'MD' for mean difference, 'SMD' for standardized mean difference
        """
        if self.outcome_type == 'binary':
            if effect_measure == 'OR':
                self._calculate_or()
            elif effect_measure == 'RR':
                self._calculate_rr()
        elif self.outcome_type == 'continuous':
            if effect_measure == 'MD':
                self._calculate_md()
            elif effect_measure == 'SMD':
                self._calculate_smd()
        
        self.effect_measure = effect_measure
        
    def _calculate_or(self):
        """Calculate odds ratios and standard errors"""
        df = self.data.copy()
        
        # Add continuity correction for zero events
        df['events1_adj'] = df['events1'] + 0.5
        df['events2_adj'] = df['events2'] + 0.5
        df['n1_adj'] = df['n1'] + 1
        df['n2_adj'] = df['n2'] + 1
        
        # Calculate odds ratios
        df['or'] = (df['events1_adj'] / (df['n1_adj'] - df['events1_adj'])) / \
                   (df['events2_adj'] / (df['n2_adj'] - df['events2_adj']))
        
        # Calculate log odds ratios and standard errors
        df['log_or'] = np.log(df['or'])
        df['se_log_or'] = np.sqrt(1/df['events1_adj'] + 1/(df['n1_adj'] - df['events1_adj']) + 
                                 1/df['events2_adj'] + 1/(df['n2_adj'] - df['events2_adj']))
        
        self.effect_sizes = df[['study', 'or', 'log_or', 'se_log_or']].copy()
        
    def _calculate_rr(self):
        """Calculate risk ratios and standard errors"""
        df = self.data.copy()
        
        # Add continuity correction
        df['events1_adj'] = df['events1'] + 0.5
        df['events2_adj'] = df['events2'] + 0.5
        df['n1_adj'] = df['n1'] + 1
        df['n2_adj'] = df['n2'] + 1
        
        # Calculate risk ratios
        df['rr'] = (df['events1_adj'] / df['n1_adj']) / (df['events2_adj'] / df['n2_adj'])
        
        # Calculate log risk ratios and standard errors
        df['log_rr'] = np.log(df['rr'])
        df['se_log_rr'] = np.sqrt(1/df['events1_adj'] - 1/df['n1_adj'] + 
                                 1/df['events2_adj'] - 1/df['n2_adj'])
        
        self.effect_sizes = df[['study', 'rr', 'log_rr', 'se_log_rr']].copy()
        
    def _calculate_md(self):
        """Calculate mean differences and standard errors"""
        df = self.data.copy()
        
        # Calculate mean differences
        df['md'] = df['mean1'] - df['mean2']
        
        # Calculate standard errors
        df['se_md'] = np.sqrt(df['sd1']**2/df['n1'] + df['sd2']**2/df['n2'])
        
        self.effect_sizes = df[['study', 'md', 'se_md']].copy()
        
    def _calculate_smd(self):
        """Calculate standardized mean differences (Cohen's d) and standard errors"""
        df = self.data.copy()
        
        # Calculate pooled standard deviation
        df['sp'] = np.sqrt(((df['n1'] - 1) * df['sd1']**2 + (df['n2'] - 1) * df['sd2']**2) / 
                           (df['n1'] + df['n2'] - 2))
        
        # Calculate standardized mean differences
        df['smd'] = (df['mean1'] - df['mean2']) / df['sp']
        
        # Calculate standard errors with Hedges' correction
        df['se_smd'] = np.sqrt((df['n1'] + df['n2']) / (df['n1'] * df['n2']) + 
                               df['smd']**2 / (2 * (df['n1'] + df['n2'])))
        
        # Apply Hedges' correction
        df['smd_hedges'] = df['smd'] * (1 - 3 / (4 * (df['n1'] + df['n2']) - 9))
        
        self.effect_sizes = df[['study', 'smd', 'smd_hedges', 'se_smd']].copy()
    
    def calculate_weights(self):
        """Calculate inverse variance weights"""
        if self.effect_sizes is None:
            raise ValueError("Effect sizes must be calculated first")
        
        # Determine which standard error column to use
        se_col = [col for col in self.effect_sizes.columns if col.startswith('se_')][0]
        
        # Calculate weights
        self.effect_sizes['weight'] = 1 / (self.effect_sizes[se_col] ** 2)
        self.weights = self.effect_sizes['weight'].values
        
    def assess_heterogeneity(self):
        """
        Assess heterogeneity using Cochrane's Q-test and I² statistic
        """
        if self.effect_sizes is None or self.weights is None:
            raise ValueError("Effect sizes and weights must be calculated first")
        
        # Determine which effect size column to use
        if self.effect_measure in ['OR', 'RR']:
            es_col = [col for col in self.effect_sizes.columns if col.startswith('log_')][0]
        else:
            es_col = [col for col in self.effect_sizes.columns if col.startswith(('md', 'smd'))][0]
        
        # Calculate weighted mean effect size
        weighted_mean = np.sum(self.effect_sizes[es_col] * self.weights) / np.sum(self.weights)
        
        # Calculate Q statistic (Cochran's Q)
        Q = np.sum(self.weights * (self.effect_sizes[es_col] - weighted_mean) ** 2)
        
        # Degrees of freedom
        df = len(self.effect_sizes) - 1
        
        # P-value for Q-test
        p_value = 1 - chi2.cdf(Q, df)
        
        # I² statistic
        I2 = max(0, (Q - df) / Q * 100) if Q > df else 0
        
        # Tau² (between-study variance)
        if Q > df:
            tau2 = (Q - df) / (np.sum(self.weights) - np.sum(self.weights**2) / np.sum(self.weights))
        else:
            tau2 = 0
        
        self.heterogeneity_stats = {
            'Q': Q,
            'df': df,
            'p_value': p_value,
            'I2': I2,
            'tau2': tau2,
            'interpretation': self._interpret_heterogeneity(I2)
        }
        
    def _interpret_heterogeneity(self, I2):
        """Interpret I² statistic according to Cochrane guidelines"""
        if I2 < 25:
            return "Low heterogeneity"
        elif I2 < 50:
            return "Moderate heterogeneity"
        elif I2 < 75:
            return "Substantial heterogeneity"
        else:
            return "Considerable heterogeneity"
    
    def fixed_effects_meta_analysis(self):
        """Perform fixed effects meta-analysis"""
        if self.effect_sizes is None or self.weights is None:
            raise ValueError("Effect sizes and weights must be calculated first")
        
        # Determine which effect size column to use
        if self.effect_measure in ['OR', 'RR']:
            es_col = [col for col in self.effect_sizes.columns if col.startswith('log_')][0]
        else:
            es_col = [col for col in self.effect_sizes.columns if col.startswith(('md', 'smd'))][0]
        
        # Calculate pooled effect size
        pooled_es = np.sum(self.effect_sizes[es_col] * self.weights) / np.sum(self.weights)
        
        # Calculate standard error
        se_pooled = np.sqrt(1 / np.sum(self.weights))
        
        # Calculate confidence interval
        ci_lower = pooled_es - 1.96 * se_pooled
        ci_upper = pooled_es + 1.96 * se_pooled
        
        # Calculate Z-score and p-value
        z_score = pooled_es / se_pooled
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        self.fixed_effect = {
            'effect_size': pooled_es,
            'se': se_pooled,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'z_score': z_score,
            'p_value': p_value
        }
    
    def random_effects_meta_analysis(self):
        """Perform random effects meta-analysis using DerSimonian-Laird method"""
        if self.heterogeneity_stats is None:
            self.assess_heterogeneity()
        
        # Determine which effect size column to use
        if self.effect_measure in ['OR', 'RR']:
            es_col = [col for col in self.effect_sizes.columns if col.startswith('log_')][0]
        else:
            es_col = [col for col in self.effect_sizes.columns if col.startswith(('md', 'smd'))][0]
        
        # Calculate random effects weights
        tau2 = self.heterogeneity_stats['tau2']
        se_col = [col for col in self.effect_sizes.columns if col.startswith('se_')][0]
        
        self.effect_sizes['weight_re'] = 1 / (self.effect_sizes[se_col]**2 + tau2)
        
        # Calculate pooled effect size
        pooled_es = np.sum(self.effect_sizes[es_col] * self.effect_sizes['weight_re']) / \
                   np.sum(self.effect_sizes['weight_re'])
        
        # Calculate standard error
        se_pooled = np.sqrt(1 / np.sum(self.effect_sizes['weight_re']))
        
        # Calculate confidence interval
        ci_lower = pooled_es - 1.96 * se_pooled
        ci_upper = pooled_es + 1.96 * se_pooled
        
        # Calculate Z-score and p-value
        z_score = pooled_es / se_pooled
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        self.random_effect = {
            'effect_size': pooled_es,
            'se': se_pooled,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'z_score': z_score,
            'p_value': p_value,
            'tau2': tau2
        }
    
    def create_forest_plot(self, figsize=(10, 8), save_path=None):
        """
        Create forest plot following Cochrane style guidelines
        """
        if self.effect_sizes is None:
            raise ValueError("Effect sizes must be calculated first")
        
        # Determine which effect size column to use
        if self.effect_measure in ['OR', 'RR']:
            es_col = [col for col in self.effect_sizes.columns if col.startswith('log_')][0]
            se_col = [col for col in self.effect_sizes.columns if col.startswith('se_')][0]
            # Convert back to original scale for display
            if self.effect_measure == 'OR':
                display_es = np.exp(self.effect_sizes[es_col])
                display_ci_lower = np.exp(self.effect_sizes[es_col] - 1.96 * self.effect_sizes[se_col])
                display_ci_upper = np.exp(self.effect_sizes[es_col] + 1.96 * self.effect_sizes[se_col])
            else:  # RR
                display_es = np.exp(self.effect_sizes[es_col])
                display_ci_lower = np.exp(self.effect_sizes[es_col] - 1.96 * self.effect_sizes[se_col])
                display_ci_upper = np.exp(self.effect_sizes[es_col] + 1.96 * self.effect_sizes[se_col])
        else:
            es_col = [col for col in self.effect_sizes.columns if col.startswith(('md', 'smd'))][0]
            se_col = [col for col in self.effect_sizes.columns if col.startswith('se_')][0]
            display_es = self.effect_sizes[es_col]
            display_ci_lower = self.effect_sizes[es_col] - 1.96 * self.effect_sizes[se_col]
            display_ci_upper = self.effect_sizes[es_col] + 1.96 * self.effect_sizes[se_col]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot individual studies
        y_positions = range(len(self.effect_sizes))
        
        # Plot confidence intervals
        for i, (_, row) in enumerate(self.effect_sizes.iterrows()):
            ax.plot([display_ci_lower.iloc[i], display_ci_upper.iloc[i]], [i, i], 
                   'k-', linewidth=1)
            ax.plot(display_es.iloc[i], i, 'ko', markersize=6)
        
        # Add pooled effect if available
        if self.random_effect:
            if self.effect_measure in ['OR', 'RR']:
                pooled_display = np.exp(self.random_effect['effect_size'])
                pooled_ci_lower = np.exp(self.random_effect['ci_lower'])
                pooled_ci_upper = np.exp(self.random_effect['ci_upper'])
            else:
                pooled_display = self.random_effect['effect_size']
                pooled_ci_lower = self.random_effect['ci_lower']
                pooled_ci_upper = self.random_effect['ci_upper']
            
            # Add diamond for pooled effect
            diamond_y = len(self.effect_sizes)
            diamond_width = pooled_ci_upper - pooled_ci_lower
            diamond_height = 0.3
            
            ax.plot([pooled_ci_lower, pooled_ci_upper], [diamond_y, diamond_y], 
                   'r-', linewidth=2)
            ax.plot(pooled_display, diamond_y, 'rs', markersize=8)
            
            # Add diamond shape
            diamond_x = [pooled_ci_lower, pooled_display, pooled_ci_upper, pooled_display, pooled_ci_lower]
            diamond_y_coords = [diamond_y, diamond_y + diamond_height/2, diamond_y, 
                              diamond_y - diamond_height/2, diamond_y]
            ax.fill(diamond_x, diamond_y_coords, 'red', alpha=0.3)
        
        # Add vertical line at null effect
        if self.effect_measure in ['OR', 'RR']:
            ax.axvline(x=1, color='gray', linestyle='--', alpha=0.7)
        else:
            ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
        
        # Customize plot
        ax.set_yticks(list(y_positions) + [len(self.effect_sizes)])
        ax.set_yticklabels(list(self.effect_sizes['study']) + ['Pooled Effect'])
        ax.set_xlabel(f'{self.effect_measure} (95% CI)')
        ax.set_title(f'Forest Plot - {self.effect_measure}')
        ax.grid(True, alpha=0.3)
        
        # Add text annotations
        if self.random_effect:
            p_text = f"p = {self.random_effect['p_value']:.3f}"
            i2_text = f"I² = {self.heterogeneity_stats['I2']:.1f}%"
            ax.text(0.02, 0.98, f"{p_text}\n{i2_text}", transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def funnel_plot(self, figsize=(8, 6), save_path=None):
        """
        Create funnel plot to assess publication bias
        """
        if self.effect_sizes is None:
            raise ValueError("Effect sizes must be calculated first")
        
        # Determine which effect size and standard error columns to use
        if self.effect_measure in ['OR', 'RR']:
            es_col = [col for col in self.effect_sizes.columns if col.startswith('log_')][0]
        else:
            es_col = [col for col in self.effect_sizes.columns if col.startswith(('md', 'smd'))][0]
        
        se_col = [col for col in self.effect_sizes.columns if col.startswith('se_')][0]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create funnel plot
        ax.scatter(self.effect_sizes[es_col], 1/self.effect_sizes[se_col], 
                  alpha=0.7, s=60)
        
        # Add vertical line at pooled effect
        if self.random_effect:
            ax.axvline(x=self.random_effect['effect_size'], color='red', 
                      linestyle='--', alpha=0.7, label='Pooled Effect')
        
        ax.set_xlabel(f'{self.effect_measure}')
        ax.set_ylabel('Precision (1/SE)')
        ax.set_title('Funnel Plot - Assessment of Publication Bias')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def sensitivity_analysis(self):
        """
        Perform leave-one-out sensitivity analysis
        """
        if self.effect_sizes is None:
            raise ValueError("Effect sizes must be calculated first")
        
        results = []
        
        for i in range(len(self.effect_sizes)):
            # Create data without study i
            temp_data = self.effect_sizes.drop(i).copy()
            
            # Recalculate weights
            se_col = [col for col in temp_data.columns if col.startswith('se_')][0]
            temp_weights = 1 / (temp_data[se_col] ** 2)
            
            # Calculate pooled effect
            es_col = [col for col in temp_data.columns if col.startswith(('log_', 'md', 'smd'))][0]
            pooled_es = np.sum(temp_data[es_col] * temp_weights) / np.sum(temp_weights)
            
            results.append({
                'excluded_study': self.effect_sizes.iloc[i]['study'],
                'pooled_effect': pooled_es
            })
        
        sensitivity_df = pd.DataFrame(results)
        
        print("Sensitivity Analysis (Leave-One-Out):")
        print("=" * 50)
        for _, row in sensitivity_df.iterrows():
            print(f"Excluding {row['excluded_study']}: {row['pooled_effect']:.4f}")
        
        return sensitivity_df
    
    def print_results(self):
        """
        Print comprehensive meta-analysis results
        """
        print("META-ANALYSIS RESULTS")
        print("=" * 50)
        print(f"Effect Measure: {self.effect_measure}")
        print(f"Number of Studies: {len(self.effect_sizes)}")
        print()
        
        # Heterogeneity results
        print("HETEROGENEITY ASSESSMENT:")
        print("-" * 30)
        print(f"Cochran's Q: {self.heterogeneity_stats['Q']:.3f}")
        print(f"Degrees of freedom: {self.heterogeneity_stats['df']}")
        print(f"P-value: {self.heterogeneity_stats['p_value']:.3f}")
        print(f"I² statistic: {self.heterogeneity_stats['I2']:.1f}%")
        print(f"Interpretation: {self.heterogeneity_stats['interpretation']}")
        print(f"Tau²: {self.heterogeneity_stats['tau2']:.4f}")
        print()
        
        # Fixed effects results
        if self.fixed_effect:
            print("FIXED EFFECTS MODEL:")
            print("-" * 20)
            if self.effect_measure in ['OR', 'RR']:
                print(f"Pooled {self.effect_measure}: {np.exp(self.fixed_effect['effect_size']):.3f}")
                print(f"95% CI: {np.exp(self.fixed_effect['ci_lower']):.3f} - {np.exp(self.fixed_effect['ci_upper']):.3f}")
            else:
                print(f"Pooled {self.effect_measure}: {self.fixed_effect['effect_size']:.3f}")
                print(f"95% CI: {self.fixed_effect['ci_lower']:.3f} - {self.fixed_effect['ci_upper']:.3f}")
            print(f"Z-score: {self.fixed_effect['z_score']:.3f}")
            print(f"P-value: {self.fixed_effect['p_value']:.3f}")
            print()
        
        # Random effects results
        if self.random_effect:
            print("RANDOM EFFECTS MODEL:")
            print("-" * 22)
            if self.effect_measure in ['OR', 'RR']:
                print(f"Pooled {self.effect_measure}: {np.exp(self.random_effect['effect_size']):.3f}")
                print(f"95% CI: {np.exp(self.random_effect['ci_lower']):.3f} - {np.exp(self.random_effect['ci_upper']):.3f}")
            else:
                print(f"Pooled {self.effect_measure}: {self.random_effect['effect_size']:.3f}")
                print(f"95% CI: {self.random_effect['ci_lower']:.3f} - {self.random_effect['ci_upper']:.3f}")
            print(f"Z-score: {self.random_effect['z_score']:.3f}")
            print(f"P-value: {self.random_effect['p_value']:.3f}")
            print(f"Tau²: {self.random_effect['tau2']:.4f}")
            print()
        
        # Interpretation
        if self.random_effect:
            if self.random_effect['p_value'] < 0.05:
                print("CONCLUSION: Statistically significant effect detected.")
            else:
                print("CONCLUSION: No statistically significant effect detected.")
            
            if self.heterogeneity_stats['I2'] > 50:
                print("NOTE: Substantial heterogeneity detected. Consider subgroup analysis or meta-regression.")
            else:
                print("NOTE: Low to moderate heterogeneity. Results are more reliable.")


# Example usage and demonstration
def example_binary_meta_analysis():
    """
    Example: Binary outcome meta-analysis (Odds Ratios)
    """
    print("EXAMPLE: Binary Outcome Meta-Analysis")
    print("=" * 50)
    
    # Example data: Studies comparing treatment vs control for mortality
    studies = ['Study A', 'Study B', 'Study C', 'Study D', 'Study E']
    n_treatment = [100, 150, 80, 120, 90]
    n_control = [100, 150, 80, 120, 90]
    deaths_treatment = [10, 15, 8, 12, 9]  # Fewer deaths in treatment group
    deaths_control = [20, 30, 16, 24, 18]
    
    # Create meta-analysis object
    meta = MetaAnalysis()
    meta.load_binary_data(studies, n_treatment, n_control, deaths_treatment, deaths_control)
    
    # Perform analysis
    meta.calculate_effect_sizes('OR')
    meta.calculate_weights()
    meta.assess_heterogeneity()
    meta.fixed_effects_meta_analysis()
    meta.random_effects_meta_analysis()
    
    # Display results
    meta.print_results()
    
    # Create visualizations
    meta.create_forest_plot()
    meta.funnel_plot()
    
    # Sensitivity analysis
    meta.sensitivity_analysis()
    
    return meta


def example_continuous_meta_analysis():
    """
    Example: Continuous outcome meta-analysis (Mean Differences)
    """
    print("\nEXAMPLE: Continuous Outcome Meta-Analysis")
    print("=" * 50)
    
    # Example data: Studies comparing treatment vs control for blood pressure reduction
    studies = ['Study A', 'Study B', 'Study C', 'Study D', 'Study E']
    n_treatment = [50, 75, 40, 60, 45]
    n_control = [50, 75, 40, 60, 45]
    mean_treatment = [-12, -15, -10, -14, -11]  # Blood pressure reduction
    mean_control = [-8, -9, -7, -10, -8]
    sd_treatment = [3, 4, 2.5, 3.5, 3.2]
    sd_control = [3.5, 4.2, 3, 3.8, 3.5]
    
    # Create meta-analysis object
    meta = MetaAnalysis()
    meta.load_continuous_data(studies, n_treatment, n_control, 
                             mean_treatment, mean_control, 
                             sd_treatment, sd_control)
    
    # Perform analysis
    meta.calculate_effect_sizes('MD')
    meta.calculate_weights()
    meta.assess_heterogeneity()
    meta.fixed_effects_meta_analysis()
    meta.random_effects_meta_analysis()
    
    # Display results
    meta.print_results()
    
    # Create visualizations
    meta.create_forest_plot()
    meta.funnel_plot()
    
    return meta


if __name__ == "__main__":
    # Run examples
    print("COCHRANE-STYLE META-ANALYSIS TOOL")
    print("=" * 50)
    print("This tool implements meta-analysis following Cochrane Handbook guidelines")
    print()
    
    # Run binary outcome example
    binary_meta = example_binary_meta_analysis()
    
    # Run continuous outcome example
    continuous_meta = example_continuous_meta_analysis()
    
    print("\n" + "=" * 50)
    print("Meta-analysis complete! Check the generated plots for visual results.")
    print("=" * 50)
