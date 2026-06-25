# citizens-bank-fraud-prediction
Machine learning pipeline for predicting check return risk using transactional and account-level features. Developed as part of the Brown University Data Science Capstone project in collaboration with Citizens Bank. All data used in this repository has been anonymized, masked, and processed to remove sensitive customer information.

## Business Example

Each row represents one current deposited check/deposit item that we want to score for return risk. For example, a customer deposits a $2,000 check into a Citizens Bank account. The check was written by a payer whose account is at Chase or Bank of America. In this setting, Citizens is the deposit bank, while the payer-side source is the drawee/payment source for the current check.

The `drawee_*` variables summarize the recent history between this same payer-side source and the Citizens account. For example, if `drawee_cnt = 4` and `drawee_sum = 5000`, the account received 4 items totaling $5,000 from this same payer-side source in the previous month. The model uses this account information, current item amount, historical transaction behavior, and payer-side history to predict whether the current check will return.
