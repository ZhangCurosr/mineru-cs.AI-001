# Catching the Rug: Early Prediction of Fraudulent Memecoins on Solana via Machine Learning

Jianghai Li

Higher School of Economics

Moscow, Russia

li.tszyankhay@hse.ru

Pavel Kuznetsov

Moscow State University

Moscow, Russia

pavelkuznetcov2002@gmail.com

Konstantin Nott-Whaley

Independent Researcher

London, UK

codingwitch@protonmail.com

Yury Yanovich

Skolkovo Institute of Science and Technology

Moscow, Russia

y.yanovich@skoltech.ru

Igor Vodolazov

Independent Researcher

Barcelona, Spain

allcryptodata@proton.me

Abstract—The rapid proliferation of memecoins on blockchain platforms has increased the risk of fraudulent activities, particularly rug pulls. While previous studies have focused on Ethereumbased tokens, this paper shifts the spotlight to Solana, the leading blockchain for memecoins by trading volume and token count. Unlike Ethereum, where rug pulls often exploit smart contract backdoors, Solana memecoin rug pulls are predominantly driven by liquidity manipulation and social dynamics. This research pioneers large-scale rug pull early detection in the Solana ecosystem by assembling a dataset of 6.4 million tokens over 7 months. Market analysis reveals that a vast majority of these memecoins exhibit rug pull characteristics within one hour of launch, highlighting the urgency of short-horizon prediction. Despite the absence of code-level features, we demonstrate that classic machine learning models, particularly Gradient Boosting (XGBoost), achieve robust performance in detecting potential rug pulls using only the first 5 minutes of trading data. Furthermore, we evaluate cross-platform generalization between PumpFun and Raydium, revealing that multi-source data fusion significantly mitigates domain shift and improves detection reliability. This study advances the understanding of DeFi fraud on highthroughput chains and provides a practical framework for protecting investors.

Index Terms—Decentralized Exchange, Scam Detection, Blockchain, Machine Learning, Deep Learning, Solana

## I. INTRODUCTION

The rapid advancement of blockchain technology, Web3 innovations, and decentralized finance (DeFi) has significantly fueled the expansion of the cryptocurrency market [1]. By May 2024, the market capitalization of cryptocurrencies surpassed 2 trillion US dollars [2], highlighting the profound influence of these technologies. To support token transactions and provide alternatives to peer-to-peer exchanges, various trading platforms and cryptocurrency exchanges have emerged. These platforms are categorized into centralized exchanges (CEXs) [3], which operate similarly to traditional financial institutions, and decentralized exchanges (DEXs) [4], which utilize smart contracts and cryptographic methods to ensure asset security.

The rise in DEX activity has significantly impacted the memecoin market. In 2024, memecoins emerged as a leading trend in the crypto world, not generating the largest trade volume overall, but driving substantial activity on DEXs [5]. Memecoins are a type of crypto asset often created as a joke or to capitalize on memes and trends on the Internet [6]. Unlike traditional cryptocurrencies such as Bitcoin or Ethereum, which have specific use cases and underlying technologies, memecoins typically lack inherent utility or serious purpose. Instead, they gain popularity through social networks, community engagement, and viral marketing. Modern memecoins are tokens deployed on established blockchains (e.g., Solana, Ethereum) rather than native coins with dedicated consensus layers. However, this growth has led to an increase in fraudulent schemes, such as rug pulls [7], [8], where developers promote a token to attract investors and then withdraw liquidity, causing the token’s value to plummet. The decentralized nature of DEXs and user anonymity further complicate the detection of such schemes. As DeFi continues to expand, developing effective methods to detect rug pulls on Solana and other blockchains is crucial to safeguard users and maintain trust in the ecosystem.

Rug pulls on decentralized exchanges pose a significant threat to investor confidence and market stability. Currently, there is a lack of specialized tools for real-time rug pull detection on DEXs, allowing fraudsters to exploit this vulnerability. Consequently, there is an urgent need for mechanisms capable of identifying potential rug pulls before they occur [9], [10].

Although artificial intelligence has become a cornerstone of financial fraud detection [11], rug pulls in decentralized memecoin markets present unique challenges: extreme class imbalance, rapid token lifecycles, and chain-specific microstructures not addressed by traditional frameworks. Recent work has begun to address these challenges on specific chains: Hu et al. [12] introduced MemeTrans, a 40k Solana memecoin dataset focused on launchpad-phase risk annotation, while Yaremus et al. [13] developed a TVL/Idle-based detection framework for the TON blockchain. Our work extends this line of research by scaling to 6.4M Solana tokens over 7 months, emphasizing early prediction and evaluating crossplatform generalization between PumpFun and Raydium.

The main contributions of this paper are summarized as follows:

• We assemble and analyze a comprehensive dataset of 6.4 million Solana memecoins over a 7-month period, exceeding prior Solana-specific studies by over 150× in scale, which enables robust statistical analysis of rare rug pull patterns.

• We conduct a rigorous evaluation of model transferability between Solana’s primary memecoin platforms, PumpFun and Raydium, revealing significant distribution shifts and demonstrating that multi-source data fusion combined with tree-based models yields the most robust crossplatform generalization.

• We validate the efficacy of short-horizon early detection, using the first 5 minutes of trading data to forecast 1- hour outcomes, specifically within the high-throughput, liquidity-driven microstructure of Solana, showing that reliable predictions can be achieved using only tabular liquidity and trading dynamics without relying on smart contract code analysis.

## II. BACKGROUND: SOLANA VS ETHEREUM MEMECOINS

Memecoins have emerged as a significant segment within the cryptocurrency landscape, characterized by their blend of humor, speculative appeal, and community engagement [14]. This section explores the contrasting community and technological ecosystems of Solana and Ethereum, two prominent blockchain platforms that support memecoins.

Solana memecoins inherently benefit from the high throughput and low transaction costs of the platform, attracting traders interested in rapid and cost-effective transactions [15]. In contrast, Ethereum memecoins leverage the platform’s established network effects and extensive user base. The Ethereum blockchain [16], which has transitioned to a proof-of-stake consensus and adopted a modular architecture, leverages established network effects. Although this approach offers flexibility and independent upgradability, it introduces complexity in layer coordination, potentially affecting user experience.

A critical issue in both ecosystems is the prevalence of rug pulls [9], a type of fraud in which developers abandon a project after securing investor funds. Ethereum’s robust DeFi infrastructure and the ease of launching tokens via the ERC-20 standard [17], [18] have made it a prime target for such scams. Ethereum memecoins, such as Shiba Inu (SHIB) and Pepe (PEPE), have gained significant traction due to community support and strategic marketing. Although smart contracts on Ethereum, primarily written in Solidity, enable decentralized, trustless, and transparent transactions, they require robust security measures such as whitelisting to manage access and reduce fraud risks.

Blockchain technology has introduced easy-to-use auditability into digital systems [19]. However, to fully benefit from this, the source code of smart contracts should be open source. While most DeFi applications adhere to this principle, many memecoins on Ethereum do not. Although the bytecode of the smart contract is always publicly available and can often be reverse-engineered [20], [21], researchers sometimes fail to find the source code in certain notable scam cases, casting a shadow on the Ethereum memecoin ecosystem.

```html
467 function fullWhitelist(address target) public
onlyOwner {
468 authorizations[target] = true;
469 isFeeExempt[target] = true;
470 isTxLimitExempt[target] = true;
471 isInternal[target] = true;
472 }
Fig. 1: Ichimoku Inu’s fullWhitelist function
```

Most of Ethereum memecoins are based on the ERC-20 token standard, though the code may have modifications. A notable example is the inclusion of whitelists. Whitelisting in smart contracts controls access to comply with regulatory requirements such as Know Your Customer (KYC) and Anti-Money Laundering (AML) regulations [22]–[24], essential for token sales and NFT minting. Implementing whitelisting involves using Solidity’s mapping to store approved addresses and restrict access to specific functions. This ensures proper access control within the contract, providing a layer of security against potential fraud.

```csv
336 function _transferFrom(address sender, address
recipient, uint256 amount) internal returns (
bool) {
337 if (inSwapAndLiquify) {
338 return _basicTransfer(sender, recipient,
amount);
339 }
340 if (!authorizations[sender] && !authorizations[
recipient]) {
341 require(tradingOpen, "");
342 }
343
344 require(amount <= _maxTxAmount ||
isTxLimitExempt[sender], "TX Limit");
345 if (isPair[recipient] && !inSwapAndLiquify &&
swapAndLiquifyEnabled && _balances[address(
this)] >= swapThreshold) {
346 marketingAndLiquidity();
347 }
348 if (!launched() && isPair[recipient]) {
349 require(_balances[sender] > 0, "");
350 launch();
351 }
352
353 // Exchange tokens
354 _balances[sender] = _balances[sender].sub(amount
"");
355
356 if (!isTxLimitExempt[recipient] &&
restrictWhales) {
357 require(_balances[recipient].add(amount) <=
_walletMax, "");
358 }
359
360 uint256 finalAmount = !isFeeExempt[sender] && !
isFeeExempt[recipient] ? extractFee(sender,
recipient, amount) : amount;
361 _balances[recipient] = _balances[recipient].add(
finalAmount);
362
363 emit Transfer(sender, recipient, finalAmount);
364 return true;
365
```

## Fig. 2: Ichimoku Inu’s \_transferFrom function

Conversely, whitelist scam tokens are designed with malicious intent, incorporating mechanisms to selectively restrict or enable trading for certain addresses [25]. Unlike legitimate tokens applying uniform rules, these scams use hidden mappings to maintain a whitelist of privileged addresses controlled by developers. These addresses are exempt from restrictions, allowing scammers to trade freely while non-whitelisted participants face barriers such as blocked transactions or excessive fees. This undermines the principles of transparency and fairness in decentralized finance.

The difference between a standard Ethereum memecoin such as Pepe Token [26] and a whitelist scam token such as Ichimoku Inu [27] lies in their smart contract logic. Pepe Token applies uniform trading rules, whereas Ichimoku Inu’s dynamic whitelisting (Figures 1 and 2) allows developers to manipulate trading conditions. Trading can be disabled for non-whitelisted addresses when tradingOpen is set to false. Whitelisted addresses are exempt from transaction limits (isTxLimitExempt), while non-whitelisted users remain subject to whale restrictions (restrictWhales). This creates unfair trading conditions that favor privileged addresses. Solana uses the Solana Program Library (SPL) as its standard for creating tokens, providing a foundation for fungible and non-fungible tokens [28]. This modular framework reduces development effort and enhances security, making scams like whitelisting abuse less feasible.

While both Solana and Ethereum offer fertile grounds for memecoin development, their distinct technological frameworks and community dynamics shape their respective ecosystems.

## III. RELATED WORK

The application of machine learning to the detection of financial fraud has been extensively surveyed [11], covering tree-based models, deep learning, and graph methods in traditional domains such as credit card fraud, loan fraud, and antimoney laundering. However, rug pulls in decentralized memecoin markets present unique challenges: extreme class imbalance, rapid token lifecycles, and chain-specific microstructures not fully addressed by traditional fraud detection frameworks.

The cryptocurrency landscape has witnessed significant developments over recent years, characterized by both innovation and deception. Fraudulent activities such as Ponzi schemes [29]–[31], pump-and-dump tactics [32], [33], and rug pulls [9], [10], [13] have become increasingly prevalent. The decentralized nature of the market facilitates these scams, with rug pulls particularly leaving investors with worthless tokens after developers withdraw liquidity. The anonymity and lack of regulation [34], [35] further complicate detection and prevention.

During the initial coin offering (ICO) era [36], a surge in fraudulent schemes emerged, reminiscent of the current memecoin trend. ICOs analysis has highlighted vulnerabilities that persist in the cryptocurrency sphere [37]. These insights are especially relevant as the modern memecoin frenzy, particularly on platforms such as Uniswap V2 [38], [39], has increased rug pull risk. Memecoins [5], [6], with their viral marketing and speculative appeal, are prime targets for fraudulent activities.

Recent advances in cryptocurrency fraud detection have focused on identifying rug pulls, especially on the Ethereum blockchain [16]. Mazorra et al. [9] developed a comprehensive labeled dataset of 26,957 tokens identified as scams on Uniswap (May 2020–September 2021), providing a theoretical classification of rug pull types–simple, sell, and trap-door– and introducing tools to identify them. Their work emphasizes token distribution metrics, such as the Herfindahl–Hirschman Index, uncovering that 90% of tokens using lock contracts like Unicrypt eventually become malicious. Srifa et al. [10] extended this to Uniswap V3, analyzing 7,450 tokens (3,212 normal, 581 rug pulls) and shifting focus from detection to timing prediction using time-series features like token volume and transaction frequency.

Beyond liquidity-based signals, behavioral pattern profiling has been explored for early warning. Cao et al. [40] constructed 12 token-level features based on three wash-trading patterns (Self, Matched, Circular) for BSC meme tokens, achieving PR-AUC = 0.9185 with Random Forest. Their work emphasizes lead-time analysis (mean 3.8 hours of early warning) and interpretable feature ablation. Although their focus is BSC and wash-trade signals, our approach targets Solana and liquidity-based labels (TVL/Idle), suggesting that combining behavioral and liquidity signals could improve cross-chain detection.

Beyond feature-based machine learning, graph neural networks have shown promise in modeling token interaction patterns. Wu et al. [41] proposed RugScreener, a temporal GNN architecture that captures dynamic transaction graphs for ERC-20 tokens on Ethereum. While their approach leverages relational structure, our work focuses on interpretable, tabular features derived from liquidity dynamics and trading behavior– offering a complementary, lower-complexity alternative suitable for real-time deployment on high-throughput chains like Solana.

Beyond Ethereum, recent work has extended rug pull detection to other blockchains. Yaremus et al. [13] developed a machine learning framework for The Open Network (TON), analyzing data from STON.fi and DeDust DEXs. Like our work, they compare two rug pull definitions–TVL-based liquidity withdrawal and idle-based trading cessation–and demonstrate that Gradient Boosting models can identify scams within five minutes of trading. Their key finding that feature distributions differ significantly across DEXs directly motivates our crossplatform analysis of Pumpfun and Raydium.

While most prior rug pull detection research targets Ethereum [9], [10], Hu et al. [12] recently released Meme-Trans, a dataset of 40k Solana memecoins with features spanning trading activity, holding concentration, and bundlelevel entity resolution. Their annotation combines statistical indicators with the detection of manipulation-patterns, achieving a 56.1% reduction in simulated financial loss. Our work complements this by scaling to 220× more tokens (6.4M vs. 40k), using liquidity-based rug pull definitions (TVL drop / prolonged idleness) rather than launchpad-phase heuristics, and evaluating cross-platform generalization between Pumpfun and Raydium.

## IV. METHODOLOGY

## A. Data Collection and Preprocessing

The data used for this research was collected directly from blockchain nodes and included transaction records, especially focusing on the minting of new tokens. Data from two decentralized exchanges (DEXes) were analyzed: Raydium and PumpFun. It is important to note that these platforms are interconnected: tokens that achieve a market capitalization of \$69,000 on PumpFun are automatically migrated to Raydium for further trading. And from November 2024 to early 2025, the crypto market witnessed Memecoin’s second explosive growth. The key event in this wave was the deep integration of AI concepts with meme culture.

For each DEX, the following data columns were extracted and standardized:
<table><tr><td colspan="2">PumpFun</td><td>Raydium</td></tr><tr><td>block_time tx_idx name url bundle_size</td><td>slot creator symbol mint gas_used amount_of_lookup_reads</td><td>slot tx_idx address signature</td></tr></table>

TABLE I: Data columns extracted from Raydium and Pump-Fun.

The data was collected over a 7-month period for each DEX (Nov. 30, 2024 – Jun. 30, 2025). Given the high transaction activity on the Solana blockchain, this timeframe provided sufficient observations for robust modeling. After obtaining the data via API, we performed preprocessing to construct a dataset of memecoin-SOL trading pairs. Table II demonstrates the scale of our token dataset and the wealth of transaction records.

<table><tr><td>DEX</td><td>Number of tokens</td><td>Total Swap Transactions</td></tr><tr><td>Raydium</td><td>97965</td><td> $5 . 9 2 3 9 5 1 \times 1 0 ^ { 8 }$ </td></tr><tr><td>PumpFun</td><td>6304235</td><td> $4 . 2 2 6 9 5 1 \times 1 0 ^ { 8 }$ </td></tr></table>

TABLE II: Dataset statistics for Raydium and PumpFun.

To effectively model token behavior, additional features were derived from the raw data, capturing both token performance on the DEXs and token creator-specific attributes.

These features were aggregated into a structured dataset to serve as input to the machine learning model. Table III lists the basic data features required for this study.

<table><tr><td>Feature</td><td>Description</td></tr><tr><td>count_tx</td><td>Total number of transactions associated with the token.</td></tr><tr><td>purchase_percentage</td><td>Percentage of transactions classified as token purchases.</td></tr><tr><td>sale_percentage</td><td>Percentage of transactions classified as token sales.</td></tr><tr><td>unique_buyers</td><td>Number of unique wallets that purchased the token.</td></tr><tr><td>unique_sellers</td><td>Number of unique wallets that sold the token.</td></tr><tr><td>total_sol_value</td><td>Total value of transactions in SOL associated with the</td></tr><tr><td>sell_sol_value</td><td>token. Total SOL value from token sale transactions.</td></tr><tr><td>buy_sol_value</td><td>Total SOL value from token purchase transactions.</td></tr><tr><td>buy_sell_cnt_ratio</td><td>Ratio of buy transaction count to sell transaction count.</td></tr><tr><td>buy_sell_value_ratio price_change_first_to_3_blocks</td><td>Ratio of buy SOL value to sell SOL value. Price change of the token during its first three blocks</td></tr><tr><td>price_change_first_to_last</td><td>of activity. Price change from first to last trade.</td></tr><tr><td>buy_price_std</td><td>Standard deviation of buy prices.</td></tr><tr><td>sell_price_std</td><td>Standard deviation of sell prices.</td></tr><tr><td>first_buy_time</td><td></td></tr><tr><td>first_sell_time</td><td>Timestamp of the first buy transaction.</td></tr><tr><td></td><td>Timestamp of the first sell transaction.</td></tr><tr><td>min_pool_info_time</td><td>Timestamp of the earliest pool information.</td></tr><tr><td>max_pool_info_time</td><td>Timestamp of the latest pool information.</td></tr><tr><td>last_trade_time</td><td>Timestamp of the last trade.</td></tr><tr><td>max_price</td><td>Maximum price.</td></tr><tr><td>min_price</td><td>Minimum price.</td></tr><tr><td>min_price_after_max</td><td>Minimum price after the maximum price.</td></tr><tr><td>first_price</td><td>Price at the first trade after mint.</td></tr></table>

TABLE III: Calculated features to observe token activity.

## B. Rug Pull Target Variable Definition

Following prior work on multi-DEX rug pull detection [13], this study adopts both Idle and TVL approaches [42], [43] to define rug pulls, each corresponding to a distinct failure mode in decentralized markets.

TVL Approach: The Total Value Locked is an important investment indicator. It is typically the total value of the LPs over a certain period of time. The higher the total value, the stronger its liquidity.

$$
\mathrm { T V L } _ { t } = \sum _ { p \in \mathcal { P } } \left( P _ { p , t } ^ { ( b a s e ) } \cdot Q _ { p , t } ^ { ( b a s e ) } + P _ { p , t } ^ { ( q u o t e ) } \cdot Q _ { p , t } ^ { ( q u o t e ) } \right)
$$

TVL rug pulls are one of the most common scams in meme investing. They typically use false advertising, promises of high returns, and marketing pump-and-dump schemes to attract users to deposit assets. At this point, TVL can experience a rapid increase in value in a short period, followed by an immediate withdrawal of funds from the liquidity pool, causing TVL to drop by 99%.

Idle Approach: The idle approach defines a rug pull in which no transactions occur for an extended period: $\begin{array} { r } { \mathrm { I d l e } _ { t } = \mathbb { I } \left( \sum _ { \tau = t - w } ^ { t } V _ { \tau } = 0 \right) } \end{array}$

When a token enters a dormant state for a long time, it effectively loses liquidity. Specifically, when the inactivity time ratio is high (e.g., close to 0.5), the market exhibits low trading intensity, and thus low liquidity.

TVL OR Idle Rug Pull: Based on the above definition, when a token triggers a TVL Rug Pull condition or becomes Idle during trading, the token is classified as a high-risk token. The specific determination rules are as follows:

The maximum drawdown depth (MDD) of TVL is defined as follows: $\begin{array} { r } { \mathrm { M D D } _ { t } = \frac { \mathrm { T V L } _ { t } - \operatorname* { m a x } _ { \tau \leq t } \mathrm { \Delta T V L } _ { \tau } } { \operatorname* { m a x } _ { \tau < t } \mathrm { \Delta T V L } _ { \tau } } } \end{array}$

This value is between $[ - 1 , \stackrel { \cdot } { 0 } ]$ . A negative value indicates that TVL has fallen from its peak, and the larger the absolute value, the more severe the liquidity withdrawal. Therefore, the Rug Pull event indicator variable is defined as follows:

$$
\mathrm { R u g } _ { t } = \mathbb { I } \left( \left( \mathrm { M D D } _ { t } < - \theta \right) \lor \left( \mathrm { I d l e } _ { t } > \Delta t \right) \right) .
$$

The thresholds θ and $\Delta t$ are empirically selected based on the distributional analysis of TVL drops and durations of inactivity.

## C. Rolling Time Series Cross-Validation

The construction of training and test sets directly impacts the reliability of model evaluation. Since Rug Pull detection is a time-dependent task, random partitioning or standard Kfold cross-validation introduces look-ahead bias, exposing the model to future market information during training and leading to overly optimistic evaluation results. Therefore, this study employs forward rolling window cross-validation based on token issuance time, reserving the last time window as an independent test set.

Another reason for using time partitioning is that Rug Pull fraud exhibits concept drift characteristics, with its onchain features constantly changing with the market. Compared to standard K-fold cross-validation, which mixes data from different time periods, a fixed-length rolling window utilizes only the most recent 3 months of data for training.

Furthermore, because Rug Pulls are a minority class event, this study implements a positive sample guarantee mechanism. When the number of positive samples in the validation set is insufficient, the validation window is automatically expanded to ensure that each validation fold contains at least 5 Rug Pull samples, thereby guaranteeing the statistical significance of the evaluation results.

This partitioning method follows the process of "historical training, future verification" to avoid data leakage and ensure that the final test set is used only for one-time evaluation, thereby improving the credibility of the experimental results.

## D. Detection Model and Parameter Optimization

The primary objective of this study is to develop a detection learning model capable of predicting token behavior. We simplified the complex task of rug pull detection into a binary classification problem, predicting a token’s future behavior using only the first 5 minutes of trading data. For traditional machine learning, we selected Random Forest, XGBoost, and MLP. We utilized Optuna (Bayesian search) to optimize hyperparameters across all folds, subsequently retraining the models with the average optimal parameters.

For transformer models, we choose FT-Transformer, Tab-Transformer, AutoInt. They represent three different ideas for tabular data modeling. We incorporated a cross-entropy loss function into our experiments to control the training and understanding of the model.

Then, we evaluated the model’s predictive performance on different platforms (Pumpfun and Raydium) under transfer training and fusion training to identify potential anisotropies in model performance.

## E. Experimental Prediction Framework

Rug Pull is defined as a fraudulent event in which the developers of a project abandon it, leading to a rapid and substantial decline in the token’s value. In this study, the target labels are binary indicators of whether a TVL rug pull or an Idle rug pull has occurred at 1 hour.

This paper constructs a basic experimental prediction framework Figure 3 for DeFi scenarios. The framework first collects unified transaction data related to PumpFun and Raydium through blockchain indexer, and performs feature engineering at the data layer to extract multi-dimensional features such as price dynamics, liquidity changes, and trading behavior. At the model layer, dedicated models for PumpFun and Raydium are designed, respectively, and a hybrid model integrating the distribution characteristics of the two types of data is further constructed to improve cross-platform generalization capabilities. In the prediction phase, new input samples are classified and discriminated, dividing tokens into potential rug pull Tokens or Good Tokens. Finally, in the evaluation module, the model performance is systematically verified using multiple indicators. The preset window of the experiment is $\mathrm { \Delta t } = 5$ min, predicting the likelihood of a rug pull within a 1-hour horizon.

![](images/299edc3c0060dcc46780535a30d1f73b06dde8fc56f8c3cdaf69b26958d50f58.jpg)  
Fig. 3: The Memecoin Future Prediction Framework uses Time t to predict $t + t _ { 0 }$

## F. Model Evaluation Metrics

In the crypto market, over 80% of Memecoins experience rug pulls. This research is based on unbalanced data from the real world of the cryptocurrency market. For unbalanced label recognition tasks, the accuracy (ACC) does not accurately reflect the performance of the classifier. When the data distribution is unbalanced, it often causes the output of the classifier to tend to the Rug class, which will have a higher classification accuracy, but performs poorly in the minority class Non-Rug.

To solve this problem, the study adopts evaluation indicators specifically designed for imbalanced classification. Rather than reporting all standard metrics, we focus on the three most robust indicators for detecting the minority class (Rug Pulls):

the F1-score for the positive class, the Matthews Correlation Coefficient (MCC), and the Area Under the Precision-Recall Curve (AUCPRC).

• F1-score (Positive Class) $\begin{array} { r l r } { = } & { { } 2 } & { \cdot \ \frac { \mathrm { R e c a l l } \times \mathrm { P r e c i s i o n } } { \mathrm { R e c a l l } + \mathrm { P r e c i s i o n } } . } \end{array}$ where $\begin{array} { r } { \mathrm { R e c a l l } = { \frac { T P } { T P + F N } } \ \mathrm { a n d \ P r e c i s i o n } = { \frac { T P ^ { * } } { T P + F P } } } \end{array}$

$\mathbf { A U C P R C } \Breve { = } \mathbf { A r e a }$ under the Precision-Recall (PR) curve

These three metrics together provide a comprehensive and unbiased evaluation framework for the severe class imbalance present in rug pull detection.

## V. NUMERICAL EXPERIMENTS

## A. Data Analysis

The data contains addresses for 6.4 million Memecoins. For all Memecoins, the study collected transaction data from the first hour for preprocessing, and Table IV illustrates some basic data characteristics. This study used a data fusion method, ensuring that both the original and fused data contained valid transaction IDs.

<table><tr><td>Metric</td><td>Pumpfun (5mins)</td><td>Raydium (5mins)</td></tr><tr><td>Total Transactions</td><td> $2 . 4 7 \times 1 0 ^ { 8 }$ </td><td> $3 . 4 2 \times 1 0 ^ { 7 }$ </td></tr><tr><td> $\operatorname { A v g } \operatorname { T x }$ </td><td> $3 . 9 3 \times 1 0 ^ { 1 }$ </td><td> $5 . 8 8 \times 1 0 ^ { 2 }$ </td></tr><tr><td>Median Tx</td><td> $1 . 2 0 \times 1 0 ^ { 1 }$ </td><td> $3 . 8 5 \times 1 0 ^ { 2 }$ </td></tr><tr><td>Total Volume (SOL)</td><td> $1 . 2 8 \times 1 0 ^ { 1 7 }$ </td><td> $3 . 0 8 \times 1 0 ^ { 1 6 }$ </td></tr><tr><td>Avg Volume</td><td> $2 . 0 4 \times 1 0 ^ { 1 0 }$ </td><td> $5 . 2 9 \times 1 0 ^ { 1 1 }$ </td></tr><tr><td>Avg Buyers</td><td> $1 . 8 3 \times 1 0 ^ { 1 }$ </td><td> $1 . 5 4 \times 1 0 ^ { 2 }$ </td></tr><tr><td>Avg Sellers</td><td> $1 . 3 0 \times 1 0 ^ { 1 }$ </td><td> $8 . 4 2 \times 1 0 ^ { 1 }$ </td></tr><tr><td>Avg Buy Ratio</td><td> $6 . 2 1 \times 1 0 ^ { - 1 }$ </td><td> $7 . 1 7 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>Avg Sell Ratio</td><td> $3 . 9 4 \times 1 0 ^ { - 1 }$ </td><td> $3 . 0 0 \times 1 0 ^ { - 1 }$ </td></tr><tr><td>Avg Price Change</td><td> $1 . 5 6 \times 1 0 ^ { 2 }$ </td><td> $5 . 0 8 \times 1 0 ^ { 5 }$ </td></tr><tr><td>Avg Buy Std</td><td> $2 . 6 2 \times 1 0 ^ { - 2 }$ </td><td> $1 . 9 9 \times 1 0 ^ { 5 }$ </td></tr><tr><td>Avg Sell Std</td><td> $4 . 9 5 \times 1 0 ^ { - 6 }$ </td><td> $6 . 2 5 \times 1 0 ^ { 2 }$ </td></tr><tr><td>Avg Last Trade time</td><td> $1 . 6 2 \times { { 1 0 } ^ { 2 } }$ </td><td> $2 . 9 7 \times 1 0 ^ { 2 }$ </td></tr></table>

TABLE IV: Descriptive statistics comparison between Pumpfun and Raydium.

From these characteristics, we can see that Pumpfun’s trading model is "multiple pools and small trades," while Raydium’s trading model is "few pools and large trades."

## B. Rug Pull Labels Analysis

Prolonged idleness indicates that a token is not being continuously or effectively traded, while a sharp TVL drop signifies that a token lacks reliable liquidity. This study extracts rug pull labels from the feature data and performs statistical analysis to facilitate a better understanding of the machine learning models’ behavior. We define a rug pull label as a token whose TVL falls below 99% or whose idle time exceeds 80% of its lifetime without trading.

The results of this research are analyzed using these rug pull labels across different DEXs. Figure 4a and Figure 4b show that these results indicate that signals based on trading idle time are particularly effective in identifying highly speculative markets such as Pumpfun, while signals based on TVL more consistently indicate liquidity withdrawals in mature markets such as Raydium, although this indication is somewhat delayed. Overall, combining these two criteria improves robustness and enables earlier detection of such behavior across different DEXs. When acquiring 5-minute data features, we drop data that has already been rug-pulled within that 5-min period. This reduces computational load and better reflects market trading patterns. Table V shows the distribution of the processed test dataset.

<table><tr><td>Metric</td><td>Rug Pull</td><td>Non Rug Pull</td></tr><tr><td>Pumpfun</td><td>43835</td><td>9711</td></tr><tr><td>Raydium</td><td>2931946</td><td>1893539</td></tr></table>

TABLE V: Label distribution of the testset between Pumpfun and Raydium.

![](images/4bd19983fb3d1a1adfcd83c586e431e118a56c7696f3f304da9edd9d14342ca2.jpg)  
(a) TVL and Idle label breakdown

![](images/1c35d21fbbf516720302df72b44d5cfff1b922e73d4bdf6c62fd40f84d8385b0.jpg)  
(b) Overall rug pull distribution  
Fig. 4: Distribution of rug pull labels on Pumpfun and Raydium.

## C. Detection Model Analysis

Statistical analysis of the labeled data reveals a significant upward trend in the rug pull risk of the Memecoin project over time. Specifically, the risk is relatively low in the early stages of trading, but the probability of liquidity withdrawal or price collapse gradually increases over time. Therefore, this study uses a 5-minute prediction window and a 1-hour alert window to evaluate the performance of different machine learning models in the rug pull prediction task.

Combining the experimental results in Table VI, it can be further observed that different data domains (such as Raydium, PumpFun, and their cross-DEX transfer) have a significant impact on model performance. In general, training within the same domain (such as Raydium or PumpFun) shows stable performance, with MCC scores around 0.25–0.36 across most models. However, cross-DEX prediction (such as Raydium→PumpFun) leads to a significant performance drop, with MCC values approaching zero or becoming negative across nearly all models, indicating substantial distributional differences between the two DEX platforms. Among the models, tree-based approaches (Random Forest and XGBoost) consistently outperform neural network-based models (MLP, FTTransformer, TabTransformer, and AutoInt) in cross-domain scenarios, demonstrating stronger robustness to domain shifts.

Interestingly, when models are trained on the fused dataset (Raydium U PumpFun) and evaluated on either domain, they achieve results comparable to or even better than single-domain training on most metrics. For instance, in the FusionßPumpFun direction, XGBoost and Random Forest achieve MCC values of 0.3947 and 0.3942, respectively, which are superior to the PumpFunßPumpFun domain-specific results (0.3562 and 0.3558). This suggests that multi-source data integration effectively improves the model’s generalization ability across DEX environments. In particular, XGBoost and Random Forest exhibit the most robust performance on AUCPRC and F1 metrics when leveraging fused data, validating the effectiveness of the data fusion strategy in the rug pull prediction task. However, it should be noted that Transformer-based models (FTTransformer, TabTransformer, AutoInt) show limited improvement from fusion on the Raydium testset (MCC values of -0.0444, -0.0543, and 0.0247, respectively), indicating that advanced neural architectures do not necessarily guarantee better cross-domain transferability in this specific financial prediction context. Overall, these findings highlight the importance of domain-aware model selection and multi-source data integration for building reliable DeFi risk prediction systems. We hypothesize that Raydium’s more mature, lower-noise trading data provides a stronger foundational signal of legitimate liquidity, which helps the model generalize better to the noisier PumpFun environment when fused.

TABLE VI: Rug Pull Prediction Results (TVL OR Idle). TVL OR Idle denotes rug pulls identified by either 99% TVL drop OR 80% idle time. Best results per column are bolded.
<table><tr><td rowspan="2">Method</td><td colspan="7">TVL OR Idle</td></tr><tr><td>R</td><td>R→P</td><td>P</td><td>P→R</td><td>R∪P</td><td>F→P</td><td>F→R</td></tr><tr><td colspan="8">Random Forest</td></tr><tr><td>F1(1)</td><td>0.7524</td><td>0.7207</td><td>0.7785</td><td>0.6106</td><td>0.7838</td><td>0.7877</td><td>0.7408</td></tr><tr><td>MCC</td><td>0.2874</td><td>-0.0434</td><td>0.3558</td><td>-0.2433</td><td>0.3938</td><td>0.3942</td><td>0.2246</td></tr><tr><td>AUCPRC</td><td>0.7578</td><td>0.5303</td><td>0.7634</td><td>0.5435</td><td>0.7897</td><td>0.7978</td><td>0.6040</td></tr><tr><td colspan="8">XGBoost</td></tr><tr><td>F1(1)</td><td>0.7532</td><td>0.7238</td><td>0.7810</td><td>0.3778</td><td>0.7833</td><td>0.7885</td><td>0.7288</td></tr><tr><td>MCC</td><td>0.2914</td><td>-0.0026</td><td>0.3562</td><td>-0.0664</td><td>0.3888</td><td>0.3947</td><td>0.1543</td></tr><tr><td>AUCPRC</td><td>0.7548</td><td>0.5771</td><td>0.7564</td><td>0.5792</td><td>0.8003</td><td>0.8011</td><td>0.6426</td></tr><tr><td colspan="8">MLP</td></tr><tr><td>F1(1)</td><td>0.7482</td><td>0.7221</td><td>0.7800</td><td>0.7068</td><td>0.7333</td><td>0.7362</td><td>0.6916</td></tr><tr><td>MCC</td><td>0.2663</td><td>-0.0427</td><td>0.3517</td><td>0.0865</td><td>0.3143</td><td>0.3144</td><td>-0.1121</td></tr><tr><td>AUCPRC</td><td>0.7379</td><td>0.5019</td><td>0.7457</td><td>0.5317</td><td>0.7437</td><td>0.7588</td><td>0.5833</td></tr><tr><td colspan="8">FTTransformer</td></tr><tr><td>F1(1)</td><td>0.7510</td><td>0.7247</td><td>0.7796</td><td>0.7130</td><td>0.7674</td><td>0.7715</td><td>0.7174</td></tr><tr><td>MCC</td><td>0.2807</td><td>-0.0508</td><td>0.3649</td><td>0.0387</td><td>0.3232</td><td>0.3257</td><td>-0.0444</td></tr><tr><td>AUCPRC</td><td>0.7506</td><td>0.4654</td><td>0.7428</td><td>0.5576</td><td>0.7297</td><td>0.7470</td><td>0.5474</td></tr><tr><td colspan="8">TabTransformer</td></tr><tr><td>F1(1)</td><td>0.6286</td><td>0.7408</td><td>0.7621</td><td>0.3530</td><td>0.7480</td><td>0.7543</td><td>0.6295</td></tr><tr><td>MCC</td><td>0.2508</td><td>0.0991</td><td>0.2867</td><td>-0.1479</td><td>0.2580</td><td>0.2719</td><td>-0.0543</td></tr><tr><td>AUCPRC</td><td>0.6997</td><td>0.6628</td><td>0.7168</td><td>0.5224</td><td>0.7003</td><td>0.7093</td><td>0.5686</td></tr><tr><td colspan="8">AutoInt</td></tr><tr><td>F1(1)</td><td>0.7501</td><td>0.7294</td><td>0.7797</td><td>0.6992</td><td>0.7694</td><td>0.7692</td><td>0.6851</td></tr><tr><td>MCC</td><td>0.2758</td><td>-0.0575</td><td>0.3491</td><td>-0.0946</td><td>0.3339</td><td>0.3307</td><td>0.0247</td></tr><tr><td>AUCPRC</td><td>0.7346</td><td>0.4775</td><td>0.7492</td><td>0.5137</td><td>0.7249</td><td>0.7354</td><td>0.5854</td></tr></table>

Note: R = Raydium (train/test on Raydium), P = PumpFun (train/test on PumpFun), F = Fusion (train on R ∪ P), → = transfer learning direction, ∪ = combined dataset

## VI. LIMITATIONS AND DISCUSSION

This study should be considered as a baseline exploration of rug pull prediction in decentralized exchanges. Although several models achieve relatively strong performance (e.g.,

AUCPRC exceeding 0.8 in certain settings), these results are not yet sufficient for real-world deployment, especially in high-risk financial environments where false negatives carry substantial investor losses.

Our results reveal substantial performance degradation under cross-DeFi settings (Table VI), indicating distribution shift between platforms. This aligns with the findings of Yaremus et al. on TON [13], where cross-DEX transfer learning also suffered from feature distribution mismatch. The pattern suggests that platform-aware adaptation–rather than naive data fusion– may be necessary for robust cross-DEX rug pull detection.

Our dataset (6.4M tokens) substantially exceeds Meme-Trans [12] (40k tokens), allowing a more robust statistical analysis of rare rug pull patterns. However, MemeTrans provides richer bundle-level features for entity resolution, suggesting a promising direction for future feature engineering. Integrating bundle detection could help identify coordinated manipulation that our current feature set may miss.

The current study relies on liquidity dynamics and price volatility features (Table III), reflecting Solana’s distinct market microstructure. Cao et al. [40] demonstrate that tradelevel wash-trading features are primary drivers of detection performance in BSC. Future work could fuse both signal types–liquidity-based and behavioral–for more robust crosschain generalization.

We employ standard tabular models, which may not fully capture the temporal dynamics, liquidity evolution, and microstructure patterns inherent in DeFi markets. More advanced modeling techniques—such as temporal GNNs [41] or sequence-based transformer architectures—could better represent these complex dynamics, albeit at a higher computational cost.

## VII. CONCLUSION AND FUTURE WORK

This paper presents the first large-scale systematic study of rug pull prediction on Solana memecoins, analyzing 6.4M tokens. We assemble the largest Solana memecoin dataset to date, exceeding prior work by 200× [12]. We implement and compare two rug pull definitions–TVL-based liquidity withdrawal and Idle-based trading cessation–within a unified study, following the methodology validated on TON [13]. We evaluate the generalization of the model between Pump.fun and Raydium, revealing a significant distribution shift that challenges naive data fusion approaches. We demonstrate that Gradient Boosting models can identify rug pulls within 5 minutes of trading, with XGBoost achieving the highest crossplatform generalization on fused data.

The experimental results highlight that different platforms exhibit distinct trading characteristics: PumpFun is characterized by high-frequency, small-volume trades, whereas Raydium tends to involve lower-frequency, large-volume trades. These structural differences lead to varying data distributions and directly affect model performance.

Data fusion proved feasible and promising, as the combination of data from multiple platforms improved model robustness and partially mitigated domain-specific biases. This suggests the potential for developing more generalizable crossplatform prediction models.

For future work, several directions are worth exploring. First, advanced architectures such as temporal GNNs [41] could be introduced to capture dynamic transaction patterns. Second, feature enrichment could integrate bundle-level entity resolution [12] and wash-trading patterns [40] with our liquidity-based features. Third, shorter prediction horizons below 5 minutes could be investigated for earlier warning, balancing accuracy against actionable lead time. Finally, realworld deployment could involve collaboration with DEX platforms to validate detection systems in production environments and measure actual investor loss prevention.

As Solana continues to dominate the memecoin market [5], robust fraud detection mechanisms become increasingly vital. This work provides a foundation for protecting investors and maintaining trust in the Solana DeFi ecosystem.

## REFERENCES

[1] E. Meyer, I. M. Welpe, and P. Sandner, “Decentralized Finance—A systematic literature review and research directions,” in ECIS 2022 Research Papers. Elsevier BV, 2021, pp. 1–25.

[2] CoinMarketCap, “Today’s Cryptocurrency Prices by Market Cap,” 2024. [Online]. Available: https://coinmarketcap.com/

[3] S. A. Lee and G. Milunovich, “Digital exchange attributes and the risk of closure,” Blockchain: Research and Applications, vol. 4, no. 2, p. 100131, 6 2023.

[4] S. Dos Santos, J. Singh, R. K. Thulasiram, S. Kamali, L. Sirico, and L. Loud, “A New Era of Blockchain-Powered Decentralized Finance (DeFi) - A Review,” in 2022 IEEE 46th Annual Computers, Software, and Applications Conference (COMPSAC). IEEE, 6 2022, pp. 1286– 1292.

[5] CMC Research, “According to CMC 2024 H1,” Coinmarketcap, Tech. Rep., 2024. [Online]. Available: https://s3.coinmarketcap.com/uploads/ according-to-cmc-2024-h1.pdf

[6] A. Stencel, “What is a meme coin? Dogecoin to the moon!” 12 2023. [Online]. Available: https://hal.science/hal-04360574https: //hal.science/hal-04360574/document

[7] Chainalysis, “The Chainalysis 2024 Crypto Crime Report,” Chainalysis, Tech. Rep., 2024. [Online]. Available: https://go.chainalysis.com/ crypto-crime-2024.html

[8] Y. Zhou, J. Sun, F. Ma, Y. Chen, Z. Yan, and Y. Jiang, “Stop Pulling my Rug: Exposing Rug Pull Risks in Crypto Token to Investors,” in Proceedings of the 46th International Conference on Software Engineering: Software Engineering in Practice. New York, NY, USA: ACM, 4 2024, pp. 228–239.

[9] B. Mazorra, V. Adan, and V. Daza, “Do Not Rug on Me: Leveraging Machine Learning Techniques for Automated Scam Detection,” Mathematics, vol. 10, no. 6, p. 949, 3 2022.

[10] S. Srifa, Y. Yanovich, R. Vasilyev, T. Rupasinghe, and V. Amelin, “Rug pull detection on decentralized exchange using transaction data,” Blockchain: Research and Applications, vol. 6, no. 3, p. 100275, 2025.

[11] H. Yang, Z. Shukur, and S. Sahran, “A review of artificial intelligence for financial fraud detection,” Applied Sciences, vol. 16, no. 4, p. 1931, 2026.

[12] S. Hu, S. F. Tekin, Y. Xu, and L. Liu, “Memetrans: A dataset for detecting high-risk memecoin launches on solana,” arXiv preprint, 2026. [Online]. Available: https://arxiv.org/abs/2602.13480

[13] D. Yaremus, J. Li, A. Kalacheva, I. Vodolazov, and Y. Yanovich, “Detecting rug pulls in decentralized exchanges: Machine learning evidence from the ton blockchain,” arXiv preprint, 2025. [Online]. Available: https://arxiv.org/abs/2509.01168v1

[14] I. Yousaf, L. Pham, and J. W. Goodell, “The connectedness between meme tokens, meme stocks, and other asset classes: Evidence from a quantile connectedness approach,” Journal of International Financial Markets, Institutions and Money, vol. 82, p. 101694, 1 2023.

[15] A. Yakovenko, “Solana: A new architecture for a high performance blockchain,” p. 32, 2018. [Online]. Available: https://solana.com/ solana-whitepaper.pdf

[16] V. Buterin, “Ethereum White Paper: A Next Generation Smart Contract & Decentralized Application Platform,” Ethereum, no. January, pp. 1–36, 2014. [Online]. Available: https://github.com/ethereum/wiki/wiki/ White-Paper

[17] F. Vogelsteller and V. Buterin, “EIP-20: ERC-20 Token Standard,” 2015. [Online]. Available: https://github.com/ethereum/EIPs/blob/ master/EIPS/eip-20.md

[18] M. di Angelo and G. Salzer, “Tokens, Types, and Standards: Identification and Utilization in Ethereum,” in 2020 IEEE International Conference on Decentralized Applications and Infrastructures (DAPPS). IEEE, 8 2020, pp. 1–10.

[19] Bitfury Group, “On Blockchain Auditability,” bitfury.com, pp. 1– 40, 2016. [Online]. Available: https://bitfury.com/content/downloads/ bitfury\_white\_paper\_on\_blockchain\_auditability.pdf

[20] M. Di Angelo and G. Salzer, “Identification of token contracts on Ethereum: standard compliance and beyond,” International Journal of Data Science and Analytics, vol. 16, no. 3, pp. 333–352, 9 2023.

[21] M. D. Angelo and G. Salzer, “Bytecode Skeletons for Sample Selection in the Analysis of Blockchain Programs,” in 2024 IEEE International Conference on Blockchain and Cryptocurrency (ICBC). IEEE, 5 2024, pp. 531–539.

[22] D. Ermilov, M. Panov, and Y. Yanovich, “Automatic Bitcoin Address Clustering,” in 2017 16th IEEE International Conference on Machine Learning and Applications (ICMLA). IEEE, 12 2017, pp. 461–466.

[23] M. Weber, G. Domeniconi, J. Chen, D. K. I. Weidele, C. Bellei, T. Robinson, and C. E. Leiserson, “Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics,” arXiv, 7 2019. [Online]. Available: http: //arxiv.org/abs/1908.02591

[24] C. Wronka, “Money laundering through cryptocurrencies - analysis of the phenomenon and appropriate prevention measures,” Journal of Money Laundering Control, vol. 25, no. 1, pp. 79–94, 1 2022.

[25] KoinX, “Crypto Whitelist Scams: How To Avoid Fund Theft When Whitelisting Addresses,” 2024. [Online]. Available: https: //www.koinx.com/blog/crypto-whitelist-scams

[26] Etherscan, “Pepe (PEPE) smart contract source code,” 2023. [Online]. Available: https://etherscan.io/address/ 0x6982508145454ce325ddbe47a25d4ec3d2311933#code

[27] “Ichimoku Inu (MOKU) smart contract source code,” 2022. [Online]. Available: https://etherscan.io/address/ 0x42AFD0Bdc9Ce209652dEa4228cFfcC6c6fBE454e#code

[28] Solana, “Solana Program Library Docs: Token Program.” [Online]. Available: https://spl.solana.com/token

[29] W. Chen, Z. Zheng, J. Cui, E. Ngai, P. Zheng, and Y. Zhou, “Detecting Ponzi Schemes on Ethereum,” in Proceedings of the 2018 World Wide Web Conference on World Wide Web - WWW ’18. New York, New York, USA: ACM Press, 2018, pp. 1409–1418.

[30] W. Chen, Z. Zheng, E. C.-H. Ngai, P. Zheng, and Y. Zhou, “Exploiting Blockchain Data to Detect Smart Ponzi Schemes on Ethereum,” IEEE Access, vol. 7, pp. 37 575–37 586, 2019.

[31] L. Galletta and F. Pinelli, “Sharpening Ponzi Schemes Detection on Ethereum with Machine Learning,” 1 2023.

[32] V. Chadalapaka, K. Chang, G. Mahajan, and A. Vasil, “Crypto Pump and Dump Detection via Deep Learning Techniques,” 5 2022.

[33] A. S. Bello, J. Schneider, and R. Di Pietro, “LLD: A Low Latency Detection Solution to Thwart Cryptocurrency Pump and Dumps,” in 2023 IEEE International Conference on Blockchain and Cryptocurrency (ICBC). IEEE, 5 2023, pp. 1–9.

[34] B. D. Feinstein and K. Werbach, “The Impact of Cryptocurrency Regulation on Trading Markets,” SSRN Electronic Journal, 8 2021. [Online]. Available: https://papers.ssrn.com/abstract=3649475

[35] Y. Chaleenutthawut, V. Davydov, M. Evdokimov, S. Kasemsuk, S. Kruglik, G. Melnikov, and Y. Yanovich, “Loan Portfolio Dataset From MakerDAO Blockchain Project,” IEEE Access, vol. 12, pp. 24 843– 24 854, 2024.

[36] G. Fenu, L. Marchesi, M. Marchesi, and R. Tonelli, “The ICO phenomenon and its relationships with ethereum smart contract environment,” in 2018 IEEE 1st International Workshop on Blockchain Oriented Software Engineering, IWBOSE 2018 - Proceedings, vol. 2018-Jan. IEEE, 3 2018, pp. 1–7.

[37] S. T. Howell, M. Niessner, and D. Yermack, “Initial Coin Offerings: Financing Growth with Cryptocurrency Token Sales,” The Review of Financial Studies, vol. 33, no. 9, pp. 3925–3974, 9 2020.

[38] H. Adams, “Uniswap Whitepaper,” 2018. [Online]. Available: https: //hackmd.io/@HaydenAdams/HJ9jLsfTz

[39] H. Adams, N. Zinsmeister, and D. Robinson, “Uniswap v2 Core,” 2020. [Online]. Available: https://uniswap.org/whitepaper.pdf

[40] D. Cao, B. Jiao, J. Yang, Y. Zhong, and W. Yang, “Early rug pull warning for bsc meme tokens via multi-granularity wash-trading pattern profiling,” arXiv preprint, 2026.

[41] C. Wu, H. Cao, J. Chen, X. Yan, G. Xu, Z. Zhao, Y. Liu, and H. Jiang, “Rugscreener: Leveraging temporal graph neural network for rugpull detection in defi,” IEEE Transactions on Information Forensics and Security, vol. 20, pp. 11 120–11 133, 2025.

[42] A. Kalacheva, P. Kuznetsov, I. Vodolazov, and Y. Yanovich, “Detecting rug pulls in decentralized exchanges: The rise of meme coins,” Blockchain: Research and Applications, p. 100336, 2025.

[43] B. Mazorra, V. Adan, and V. Daza, “Do not rug on me: Leveraging machine learning techniques for automated scam detection,” Mathematics, vol. 10, no. 6, p. 949, 2022.