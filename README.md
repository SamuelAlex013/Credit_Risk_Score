# Compound Wallet Risk Scoring Assignment

## Overview
This project implements a machine learning-based risk scoring system for Ethereum wallets, focusing on their activity with the Compound V2/V3 lending protocol. The system is designed to:
- Process on-chain transaction data for a list of wallet addresses
- Engineer features that reflect each wallet's DeFi risk profile
- Assign a risk score (0-1000) to each wallet using a trained ML model
- Output results in a CSV format

## Assignment Prompt
Given a list of wallet addresses, your tasks are:
1. **Fetch Transaction History:** Retrieve transaction data for each wallet from Compound V2 or V3 protocol
2. **Data Preparation:** Organize and preprocess the data to create meaningful features for risk analysis
3. **Risk Scoring:** Develop a model to assign each wallet a risk score (0-1000), with clear documentation of feature selection, normalization, and scoring logic

### Deliverables
- A CSV file with columns:
  | wallet_id | score |
  |-----------|-------|
  | 0xfaa0768bde629806739c3a4620656c5d26f44ef2 | 732 |
- A brief explanation of:
  - Data collection method
  - Feature selection rationale
  - Scoring method
  - Justification of risk indicators

## Solution Structure

### Data Collection
- **Source:** On-chain transaction data is fetched for each wallet address, focusing on interactions with Compound protocol contracts (V2/V3 markets and Comptroller).
- **Storage:** Each wallet's transactions are saved as `{wallet}.json` in the `transactions/` directory.
- **Filtering:** Only transactions with Compound protocol addresses are used for DeFi risk analysis.

### Feature Engineering
- **Feature Set:** 20+ features are extracted per wallet, including:
  - Health factor estimation (volatility, value patterns)
  - Liquidation history
  - Borrowing and supply ratios
  - Transaction frequency and timing volatility
  - Success/failure rates
  - Portfolio diversity (unique counterparties)
  - Gas usage and efficiency
  - Activity recency and consistency
  - Concentration and trend metrics
- **Normalization:** Features are log-transformed, scaled, or ratioed as appropriate to ensure comparability and robust ML performance.

### Risk Scoring Model
- **Model:** RandomForestRegressor (scikit-learn pipeline with StandardScaler)
- **Training:** Model is trained on 5000+ synthetic samples with realistic feature/risk patterns, ensuring broad coverage and generalization
- **Scoring:**
  - Wallets with Compound activity are scored by the ML model
  - Wallets with no Compound activity are penalized with higher risk scores (750-980)
  - Wallets with no transaction data are assigned maximum risk (950+)
- **Output:** Results are saved to `wallet_risk_scores.csv` with columns:
  - `wallet_address` (or `wallet_id`)
  - `risk_score` (0-1000, higher = riskier)

### Justification of Risk Indicators
- **DeFi Activity:** Active, successful Compound users are lower risk; inactivity or no DeFi = higher risk
- **Volatility & Consistency:** High volatility, inconsistent repayments, or failed transactions increase risk
- **Recency:** Long inactivity or few transactions signals higher risk
- **Portfolio Diversity:** More counterparties and diverse activity reduce risk
- **Gas & Efficiency:** Inefficient or failed transactions are riskier

## Usage
1. Place all wallet transaction files in the `transactions/` folder, named `{wallet}.json`
2. Ensure `Wallet.csv` contains a `wallet_id` column with all wallet addresses
3. Run the script:
   ```bash
   python wallet_risk_analyzer.py
   ```
4. Results will be in `wallet_risk_scores.csv`

Install dependencies:
```bash
pip install -r requirements.txt
```

## Output Example
| wallet_id | score |
|-----------|-------|
| 0xfaa0768bde629806739c3a4620656c5d26f44ef2 | 732 |

## Notes
- Only Compound protocol DeFi activity is considered for risk scoring
- Wallets with no Compound transactions are penalized with higher risk
- The ML model is trained on synthetic data for robust, realistic scoring
