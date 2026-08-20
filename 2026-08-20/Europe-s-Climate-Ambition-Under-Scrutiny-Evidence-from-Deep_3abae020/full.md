# Europe’s Climate Ambition Under Scrutiny: Evidence from Deep Learning Emission Projections

Jacopo Ghirri<sup>1,2,3,</sup> <sup>\*</sup>, Carlos Rodriguez-Pardo<sup>1,2,3</sup>, Lara Aleluia Reis<sup>2,3,4</sup>, and Massimo Tavoni<sup>1,2,3</sup>

<sup>1</sup>Politecnico di Milano, Piazza Leonardo da Vinci 32, Milan, 20133, Italy

<sup>2</sup>CMCC Foundation - Euro-Mediterranean Center on Climate Change, Via Marco Biagi 5, Lecce, 73100, Italy <sup>3</sup>RFF-CMCC European Institute on Economics and the Environment, Via Bergognone 34, Milan, 20144, Italy <sup>4</sup>CENSE—Center for Environmental and Sustainability Research & CHANGE—Global Change and Sustainability Institute, NOVA School of Science and Technology, NOVA University Lisbon, Caparica, 2829-516, Portugal <sup>\*</sup>Corresponding Author. Email: jacopo.ghirri@polimi.it

## Abstract

The European Union has committed to reducing greenhouse gas emissions 55% below 1990 levels by 2030, but whether current trends are compatible with this ambition remains uncertain. We apply deep learning to high-resolution socioeconomic and sectoral data across EU27 member states till 2023 to project sectoral $\mathrm { C O } _ { 2 }$ trajectories under current trends, extrapolating observed sectoral momentum without assuming changes in the pace or efectiveness of the policy environment beyond what is already reflected in historical data. We project that EU27 emissions will exceed the 2030 target by 35% (620 Mt CO shortfall), with only a small minority of countries on trajectories consistent with the bloc’s commitments. While the Power sector achieves target-consistent reductions driven by the renewable transition, Mobility shows minimal progress and accounts for over a third of total emissions by 2030, reflecting a structural inertia across member states rather than geographically concentrated lag. Our findings indicate that substantial additional intervention is required to close Europe’s ambition-implementation gap, and call for establishing up-to-date energy information in Europe.

## 1 Introduction

Accurately forecasting short-term greenhouse gas emissions is key to assessing climate progress and reorienting public policy towards efective mitigation strategies. In the European Union (EU), understanding the trajectory of sectoral emissions is essential for assessing whether the EU remains on course toward its 55% reduction objective in 2030 [25, 32], but also for identifying where progress is lagging and where additional policy action is needed. CO emissions are determined by a high-dimensional mix of economic activities, energy use and consumption patterns, mobility demand, technological change, and climate variability [15], factors that evolve heterogeneously across countries and sectors. Capturing these dynamics in a systematic, data-driven, and interpretable way remains a major challenge [8, 26], rooted in the limitations of both structural and machine-learning approaches to emissions modeling.

Structural models such as Integrated Assessment Models (IAMs), macroeconomic frameworks, and energy system optimization models are designed to evaluate energy and climate policies under internally consistent assumptions about the functioning of the energy system [33], making them well-suited for scenario analysis but less adequate for accurate short-term forecasting. In practice, imperfect information, limited access to finance, institutional rigidities, and heterogeneous objectives can substantially delay or attenuate policy responses [13, 14], and the realized short-term impact of carbon pricing or commandand-control instruments frequently diverges from model-based projections [2, 4]. Beyond this, IAMs and energy system optimization frameworks typically assume rational actors, perfect policy implementation, and a degree of intertemporal foresight that is rarely observed in practice [13, 14, 4]: empirical trend extrapolation provides a complementary lens precisely because it makes no such assumptions, grounding projections in the behavioral and structural patterns that have actually governed emission dynamics.

The availability of rich, sectoral-scale datasets has expanded substantially in recent years. Indicators compiled by Eurostat, the IEA, FAO, and Copernicus now cover a broad range of economic, energy, land-use, and climate variables at annual resolution across European countries. These data provide an unprecedented opportunity to build models that move beyond stylized scenarios and instead reflect empirical sectoral patterns. Despite these developments, existing forecasting tools have been slow to integrate such heterogeneous signals, and most approaches still rely on simplified assumptions or coarse representations that mask sector-specific drivers, though recent contributions suggest this gap is beginning to close [39, 40, 15, 44].

Existing approaches span a spectrum from interpretable white-box models to opaque black-box methods [15, 8, 26]. On one side of the spectrum, white-box models span a range from standard econometric regressions [6, 43, 45, 46] to structural macro–energy frameworks [33]. Econometric approaches can estimate historical relationships between emissions and their drivers, but typically rely on low-dimensional specifications that aggregate across sectors or assume stable functional forms, limiting their ability to capture the heterogeneous, nonlinear dynamics that characterize real emission trajectories [6, 2, 21, 43, 45, 46]. On the other extreme, black-box machine-learning models have achieved high predictive accuracy, but at the expense of usability for policy insight. Autoregressive neural networks or models regressing directly on GDP, energy demand, and past emissions can fit historical data well [38, 41, 28, 27], yet their internal representations are not always directly mappable to physical or economic drivers, limiting their value for diagnosing why emissions change or for understanding which structural factors are associated with emission dynamics [15].

These approaches thus leave a specific and policy-relevant question unanswered: given observed sectoral dynamics, what emissions trajectory is Europe currently on if current trends persist?

Answering this question also requires confronting a persistent limitation shared by both modeling families: the treatment of uncertainty as an add-on rather than a core modeling principle [1, 26]. While uncertainty quantification has recently begun to attract attention in the climate economics field [23, 12], most data-driven forecasts remain deterministic, and most scenario-based models address uncertainty through discrete scenario comparison rather than probabilistic characterization, leaving statistical variability and structural ambiguity poorly understood and accounted for [30, 19, 42, 22]. This tendency toward deterministic projection often leaves the range of plausible outcomes undercharacterized [29]. In scenario-based assessments, this limitation is compounded by the fact that projected pathways also depend on normative assumptions about policy ambition, technology deployment, and feasibility.

Emissions trajectories are determined by shocks, structural transitions, and policy changes that are deployed unevenly across time, space, and sectors. Models not accounting for such a structural and deep uncertainty risk overstating confidence in their predictions [16], and are more prone to overfitting and fragile extrapolation. For climate policy, where decisions must remain robust under a wide range of plausible futures and where the cost of underestimating uncertainty can be severe, characterizing the spread of possible outcomes is as important as predicting central trends.

Here, rather than specifying assumptions about future policy efectiveness, technology adoption rates, or demand trajectories, we track high-frequency sectoral indicators to assess whether the current policy portfolio is translating into measurable emission reductions consistent with stated goals. The resulting projections are best understood as a current-trends counterfactual, the trajectory Europe is on absent additional intervention, which complements rather than competes with scenario-based assessments of what policies could deliver.

## 2 Framework Overview

We forecast sectoral $\mathrm { C O } _ { 2 }$ emissions across EU27 member states from high-dimensional observational data spanning multiple indicators across energy production and consumption, fuel prices, electric vehicle sales, renewable capacity, land use, and international trade, among others (full list in Table 1). $\mathrm { C O } _ { 2 }$ is modeled as a proxy for total greenhouse gas emissions; non- $\mathbf { C O } _ { 2 }$ gases such as methane and nitrous oxide are excluded as their sectoral dynamics are governed by distinct and more complex processes that fall outside the scope of this study. Given the high dimensionality of these inputs, we compress them into a structured latent representation using a variational autoencoder, which captures the core transition dynamics of each country-year while preserving natural variability. This latent representation, together with context variables such as GDP, population, and climate conditions for which reliable near-term projections exist, is then used to forecast emissions autoregressively through 2030. Uncertainty is propagated explicitly throughout, from the probabilistic latent encoding through to the final sectoral emission estimates.

We model $\mathrm { C O } _ { 2 }$ emissions as a function of high-dimensional, heterogeneous socioeconomic and sectoral indicators observed across EU27 member states. These indicators span diferent thematic domains: energy consumption and prices, electricity generation and trade, transportation of goods and people, economic indicators, land use and agriculture, and climate. The full set of variables, together with their sources, is reported in Table 1. All indicators are observed annually from 2010 through 2023; 2024 sectoral $\mathrm { C O } _ { 2 }$ emissions are available from Eurostat and used as an anchor for the projection chain, but the broader indicator panel for that year is not yet suficiently complete to drive the encoder directly, a point we return to in Section 4.4.

Formally, for each country � and year �, we observe a vector of transition indicators $\boldsymbol { x } ( t , i ) \in \mathbb { R } ^ { d }$ alongside context variables $c ( t , i )$ . Our target is a vector of sector-wise $\mathrm { C O } _ { 2 }$ emissions $\boldsymbol { y } ( t , i ) \in \mathbb { R } ^ { s }$ across $s = 6$ sectors.

We identify three core challenges. First, � is large relative to the number of country-year observations, and many indicators co-move in complex, nonlinear ways. Second, we aim to project emissions forward to 2030 without conditioning on assumed policy outcomes or structural transitions. We want to extrapolate empirical momentum, not simulate the outcomes of hypothetical policy interventions. Third, both emissions trajectories and transition indicators present noise and uncertainties, and a framework not accounting for these issues risks overfitting and failing to capture the true underlying mechanism guiding decarbonization.

Our framework chains three components sequentially for each EU27 country, as reported in Figure 1. A Variational Autoencoder (VAE) [17] compresses each country-year observation $x ( t , i )$ into a low dimensionality latent space $z ( t , i ) \sim \Lambda ( \mu , \sigma ^ { 2 } )$ , acting as an internal state representation capturing both the central structure of the data and its natural variability. A latent forecaster then iterates $\boldsymbol { z } ( t , i )$ forward autoregressively from the last observed year through 2030, conditioned on context variables $c ( t , i )$ , for which there are reliable near-term projections. An emission predictor finally maps the projected latent sequence and context variable to sectoral emission estimates $\hat { y } ( t , i ) .$ , alongside a learned uncertainty vector $u ( t , i )$ that penalizes overconfident extrapolation in sectors where emissions are structurally volatile. Full architectural and training details are given in the Methods section 4.3.

## 3 Results

A note on reported uncertainty All numerical projections are reported as point estimates accompanied by 90% model consistency bands (m.c.b.), shown in brackets as [m.c.b. lower – upper]. These bands are derived from 10,000 Monte Carlo samples of the encoder’s latent distribution, post-hoc calibrated to achieve 90% coverage on historical residuals (see Methods Section 4.5). They reflect uncertainty in the mapping from observed indicators to emissions given the expected latent trajectory, and are not predictive distributions over future outcomes. In particular, they do not capture uncertainty in the latent forecasting step, future structural shocks, or policy discontinuities. They are most informative as comparative measures of relative predictability across sectors and countries, rather than as confidence bounds on realized 2030 emissions.

![](images/e52d1e6596e32c250d8deb2d901da820c418c376acac007a86ecbec62b65b1bf.jpg)  
Figure 1: Overview of the modeling framework. The framework consists of three components: a VAE that maps high-dimensional transition indicators $x ( t , i )$ to a probabilistic latent representation $\boldsymbol { z } ( t , i ) ;$ ; a latent-space forecasting model that projects these representations forward using context variables $c ( t , i )$ ; and an emission predictor that maps the projected states and context variables to sector-wise emission estimates $\hat { y } ( t , i )$ and uncertainty outputs $u ( t , i )$ . Here � denotes the year and � the EU27 member state. For each country, observed indicators are encoded, iteratively projected to 2030, and decoded into sectoral $\mathrm { C O } _ { 2 }$ emission projections

## 3.1 Falling short of targets

We project that, if current trends persist, EU27 $\mathrm { C O } _ { 2 }$ emissions are going to decline from 2.72 Gt in 2024 to 2.36 [m.c.b. 2.19 – 2.53 Gt] Gt by 2030. A reduction of $1 3 . 2 \% [ \mathrm { m . c . b . \ } - 1 9 . 5 \% - - 7 \% ]$ , which is falling short of the 36% reduction needed to abate 55% of $1 9 9 0 \mathrm { C O } _ { 2 }$ emissions, constituting a gap of $6 2 2 \mathrm { M t } \mathrm { C O } _ { 2 }$ [m.c.b. 454 – 793 Mt] from the 1.74 Gt target. This target is derived by applying the 55% reduction to 1990 CO<sub>2</sub> emissions alone, as a proxy for the FF55 commitment, which formally applies to all greenhouse gases and does not specify a $\mathrm { C O } _ { 2 } .$ -only reduction level. Our focus on $\mathrm { C O } _ { 2 }$ reflects both its dominance in greenhouse gas emissions and the greater regularity and tractability of its sectoral dynamics relative to non- $C \mathbf { O } _ { 2 }$ gases.

Figure 2 contextualizes this finding against six alternative projection sources. Our estimate sits close to the OECD Business as Usual scenario (16% emission reduction over the same period) and substantially above the most optimistic scenario, EEA With Additional Measures (43% emission reduction). This ordering is to be expected: third-party baselines, even conservative ones, incorporate forward-looking assumptions about policy efectiveness, technology adoption rates, and future demand trajectories that go beyond what is directly observable in historical data. Our framework makes no such assumptions in either direction, neither does it anticipate policy acceleration nor assumes policy rollback. Instead, it extrapolates the empirical patterns already present in the data at their observed pace. The alignment between our estimate and the OECD BAU, derived through an entirely diferent methodology, corroborates that current momentum places Europe well short of its 2030 commitments.

![](images/07c7202e19d6442211a138edb20084aedb03cc9a85b5e3733542f84f6b575b22.jpg)  
Figure 2: Projected EU27, and all member states, $\mathrm { C O } _ { 2 }$ emission change from 2024 to 2030. Panels (a) and (b) show the projected percentage and absolute change in EU27 aggregate emissions relative to the 2024 baseline, respectively, for this study and six reference projections (OECD BAU, OECD Energy Transition, EEA With Existing Measures, EEA With Additional Measures, PyPSA Baseline, PyPSA Fit for 55). The dashed line indicates the change needed to meet a 55% reduction from 1990 levels. Panel (c) shows the distribution of projected emission changes by member state on a symmetric logarithmic scale, with each source plotted separately per country. Countries are ordered by their projected change under this study

Panel c of Figure 2 shows the distribution of projected emission changes from 2024 to 2030 across member states. Decarbonization is uneven: 10 countries show projected reductions exceeding 10%, while the other 17 show slower decarbonization, or even increases. The largest emitters, Germany, France, Italy, Poland, and Spain, collectively accounting for 65.5% of 2024 emissions, achieve a collective decrease in emissions of 15% [m.c.b. -24% – -6%], with Germany showing the strongest decline among this group.

## 3.2 Uneven progress

Figure 3 decomposes the EU27 emission trajectories from 2010 to 2030 by country and sector, revealing substantial heterogeneity in decarbonization progress beneath the aggregate trend. Total EU27 emissions declined from 3.66 Gt in 2010 to 2.72 Gt in 2024, a 25.6% reduction over the historical period. Our projections indicate a continued decline to 2.36 Gt by 2030, corresponding to a total reduction of 35.5% [m.c.b. -40.1% – -30.8%] from 2010 levels, but with markedly diferent contributions across sectors.

The Power sector stands out as the clearest decarbonization success, declining from 1.09 Gt in 2010 to a projected 0.35 Gt [m.c.b. 0.27 – 0.44 Gt] in 2030 (67% [m.c.b -74.9% – -59.9%] reduction). This trajectory is broadly consistent with the scale of reduction required under Fit for 55 targets, and reflects the structural shift toward renewable generation that has accelerated across the EU over the past decade. In contrast, Mobility is projected to remain the largest emitting sector in 2030 at 0.87 Gt [m.c.b. 0.81 – 0.93 Gt] (36.8% of total [m.c.b. 35% – 38.7%]), having declined only 7.5% from 2010 [m.c.b. -14% – -0.8%], the slowest trajectory of any sector. Industry shows a significant, but slower, decarbonization (31.4% from 2010 [m.c.b. -37.5% – -25.1%]), while Heating & Cooling, Land Use, and Other sectors contribute smaller shares (21% aggregate share of 2030 projected emissions [m.c.b. 19.7% – 22.4%]), totaling 0.5 Gt in 2030 [m.c.b. 0.45 – 0.54 Gt]. Notably, the COVID-19 pandemic produced a visible dip in emissions around 2020, particularly in Mobility, with post-pandemic recovery evident before the projected decline resumes.

Figure 4 confirms that this sectoral pattern holds broadly across member states, but with considerable heterogeneity. Power decarbonizes consistently: 11 of 27 countries show projected reductions exceeding 50%, reflecting the structural clarity of the ongoing renewable transition. The picture for Mobility is starkly diferent, as 11 countries show projected increases, with no clear regional pattern, suggesting that the resistance to transport decarbonization is structural rather than geographically concentrated. Germany and Italy, the two largest contributors to the block’s emissions, emerge as broad-based decarbonizers, achieving reductions across all six sectors. The Heating & Cooling sector displays the highest cross-country variance, ranging from 92% reduction to 72% increase. These extreme swings are however concentrated in countries that already exhibit exceptionally low per capita emissions in this sector, such as Sweden, where the small absolute baseline amplifies percentage changes. Land Use similarly shows high cross-country variance. A full country-by-sector breakdown, comprehensive of model confidence, is reported in Figure 9 in the Supplementary Information.

## 3.3 Model Interpretability

To characterize which inputs drive the model’s sectoral predictions, we compute gradient-based attributions across the encoder’s dimensionality reduction $( \partial \mu _ { j } / \partial x _ { i } )$ and the predictor’s emission attribution (��ˆ<sub>�</sub>/��<sub>�</sub>). Figure 5 shows, for each output, the top five activations by absolute gradient, averaged across all countryyear observations, and for each selected latent feature the top five input variables by gradient activation. Flows are proportional to the mean absolute gradient.

The attribution structure highlights a strong influence of context variables, particularly GDP and population, which enter the predictor directly and have broad activations across all sectors. Among the latent dimensions selected as most informative for emission predictions, there is a broad sensitivity to energy: from fuel pricing, to power generation and electricity trade. Moreover, we highlight two particular cases. Latent 0 stands out as the most informative dimension, with the electricity balance (exports, imports, and pumped storage) emerging as its dominant input. This is consistent with energy balance serving as a compressed fingerprint of each country’s structural energy position, with downstream consequences that extend across sectors. Latent 3, by contrast, is exclusively determined by the domestic generation mix, with no detectable contribution from trade or price variables, and connects strongly to power sector emissions, suggesting the VAE has isolated electricity generation as a dedicated and separable dimension of the latent space. Fossil fuel prices appear as the other major carrier of information, routing through the remaining active latent dimensions. Together, these results confirm the centrality of comprehensive energy policy, spanning generation mix, trade, and pricing, as the primary structural determinant of near-term emission trajectories across the EU.

![](images/46c04a1068153aae740877d49548ec1ab181a0173d3c20a735f702509df860ec.jpg)

![](images/73edbb658dff7f659157ab313ceff61def4575d5e66c8ec129f77a375944e7fb.jpg)

![](images/37d0c534a0165c8c8fe9dace8c0e5eeae4ed2063a2de74b91118d4738e7256c2.jpg)

![](images/b4fcca10611331133321577b9bc86fc1b2977a590506a5cf870e8b6abc7052cc.jpg)  
Figure 3: EU27 $\mathrm { C O } _ { 2 }$ emission trajectories from 2010 to 2030, decomposed by country and sector, with projected 2030 uncertainty distributions. Panels (a) and (b) show historical (2010–2024) and projected (2025–2030) stacked area charts of total $\mathrm { C O } _ { 2 }$ emissions, decomposed by country group and sector, respectively. The dashed vertical line marks the boundary between the observed and projected periods. Panels (c) and (d) show box-and-whisker plots of the 2030 projected emissions across $1 0 { , } 0 0 0$ Monte Carlo samples, by country group and sector. Boxes show the interquartile range and whiskers the 90% model-consistency band, rescaled to match model residuals ditribution (see Methods Section 4.5).

![](images/bfd4778aede9dddc46fb960393c6665b15b516f8f89551cb2251bae62075850d.jpg)  
Figure 4: Projected total and sectoral $\mathrm { C O } _ { 2 }$ emission changes from 2024 to 2030 across EU27 member states. Panel a shows the aggregate change across all sectors. Panels b to g show changes for Heating & Cooling, Industry, Land Use, Mobility, Other, and Power, respectively. Circle color encodes the direction and magnitude of change (blue: decreasing; red: increasing), with each panel using an independent color scale to reflect sector-specific ranges. Circle size encodes 2024 per capita emissions, with reference values shown below each panel.

![](images/b6f6dda7d6eb800dfcc2637d736e79e842440a2b7cfc7c4da005903d2557ac67.jpg)  
Figure 5: Gradient-based attribution flows across the emission prediction pipeline. Input variables (left) connect through VAE latent dimensions (center) to sectoral emission outputs (right). Context variables enter the predictor directly, bypassing the encoder. For each emission output, the five most influential variables are shown; for each selected latent dimension, the five most influential input variables are shown. Node and edge widths are proportional to mean absolute gradient, averaged across all country-year observations in the training sample.

## 4 Conclusions

The European Union’s 2030 climate ambition faces a substantial implementation gap. Under current trends, we project that EU27 $\mathrm { C O } _ { 2 }$ emissions are going to reach 2.36 Gt by 2030, a 13% reduction from 2024 levels that falls approximately 620 Mt $\mathrm { C O } _ { 2 }$ short of what is needed to achieve the Fit for 55 target. Our data-driven framework, which extrapolates empirical sectoral momentum without embedding policy assumptions, places Europe’s trajectory closer to a business-as-usual baseline than to any scenario consistent with its stated goals. Decarbonization progress is highly uneven across member states, with only a small minority of countries on trajectories consistent with the block’s commitments, indicating that the aggregate shortfall reflects a near-universal failure of current trends rather than the drag of a few lagging economies.

The sectoral picture is one of stark divergence. Power generation is on a structurally credible decarbonization path, projected to decline $6 7 \%$ from 2010 levels by 2030, reflecting the accelerating renewable transition across the EU. Mobility, by contrast, remains deeply resistant to change: projected to account for 36.8% of 2030 emissions with only a 7.5% reduction since 2010, it represents the single largest obstacle to European climate targets. Crucially, this resistance is not geographically concentrated but broadly structural across member states, pointing to systemic rather than country-specific failures. Electric vehicles (EVs) ofer the best and most afordable way to reduce emission from personal transportation. Our model is trained until 2023, thus incorporating the slowdown in EV sales observed by 2024. Currently, EV sales is ramping up again. It might be that these more recent trends, partly spurred by the energy crisis in the Middle East, will speed up the decarbonization of mobility, and that our estimates might be overstated, though it is worth reminding that freight emissions in Europe have been showing no sign of decline.

Our attribution analysis further suggests that energy system structure, generation mix, electricity trade balance, and fuel pricing, are the primary actionable determinants of near-term emission trajectories, pointing to these as the most plausible levers to shift current momentum.

These findings carry direct implications for European climate governance. The gap between current trajectories and stated commitments reflects the distance between the policy environment that has actually shaped emission dynamics and the more ambitious conditions assumed in optimistic scenarios. Closing this gap within the remaining timeframe will require interventions that go substantially beyond current policy portfolios, particularly in transport. Our results make clear that Europe’s stated climate ambition is not yet matched by observable decarbonization momentum, especially in some crucial sectors such as transportation. Closing this gap demands urgent, targeted action rather than reliance on favorable assumptions about future policy efectiveness. To track the evolution of the energy system against its policy objectives, it is crucial to develop a database of energy and economic statistics that can enable near-real-time assessments, leveraging deep learning methods such as those used in this paper.

## 4.1 Data Sources

Table 1 reports the data sources on which all models have been trained, and Table 2 reports the sectorial decomposition of emission sources.

## 4.2 Framework and Modeling rationale

We model sector-wise $\mathrm { C O } _ { 2 }$ emissions $\mathbf { y } ( t , i ) \in \mathbb { R } ^ { s }$ across $s = 6$ sectors as a function of high-dimensional transition indicators $\mathbf { x } ( t , i ) \in \mathbb { R } ^ { d }$ and context variables c(�, �), observed annually for each EU27 member state �. The framework chains three learned components: an encoder $q _ { \phi }$ , a latent forecaster $f _ { \theta } ,$ , and an emission predictor $g _ { \psi }$ , trained sequentially on the historical panel and applied autoregressively to project sectoral trajectories through 2030.

The high dimensionality and nonlinear co-movement of $\mathbf { x } ( t , i )$ motivate a learned latent representation rather than feature selection or standard dimensionality reduction. The encoder $q _ { \phi }$ maps $\mathbf { x } ( t , i )$ to a probabilistic latent state $\mathbf { z } ( t , i ) \sim N ( \mu _ { \phi } ( \mathbf { x } ) , \sigma _ { \phi } ^ { 2 } ( \mathbf { x } ) )$ , encoding each observation as a distribution over latent space rather than a fixed point. Unlike deterministic alternatives such as PCA [9], this probabilistic compression is robust to measurement noise and heterogeneous data quality, both pervasive features of cross-national indicator datasets, and preserves the cross-indicator covariance structure without requiring individual projections for each of the � inputs.

<table><tr><td>Category</td><td>Variables</td><td>Source</td></tr><tr><td>Emissions</td><td> $\mathrm { C O } _ { 2 }$  emissions by NACE activity and households (total and per capita)</td><td>Eurostat</td></tr><tr><td>Agriculture</td><td>Production of fruits, meats (total and poultry), vegetables, wheat</td><td>FAO</td></tr><tr><td>Energy Consumption</td><td>GJ per capita energy use (industry, transport, final consumption)</td><td>Eurostat</td></tr><tr><td>Economic Indicators</td><td>Energy taxes (€M), GDP (quarterly in €M), Population</td><td>Eurostat</td></tr><tr><td>Transportation</td><td>Modal split of freight (air, rail, road, sea), electric vehicle sales and stock, trains (diesel/electric)</td><td>Eurostat, IEA</td></tr><tr><td>Renewable Energy</td><td>Heat pumps (GWh), solar thermal collectors&#x27; surface</td><td>Eurostat</td></tr><tr><td>Land Use</td><td>Country area, coastal waters, cropland, forests</td><td>FAO</td></tr><tr><td>Electricity</td><td>Distribution losses, imports/exports, production (combustible, renewable, solar,</td><td>IEA</td></tr><tr><td>Climate</td><td>wind, hydro) Temperature (population/area weighted), rainfall, temperature variability</td><td>ERA5 (Copernicus)</td></tr><tr><td>Energy Prices</td><td>Gasoline, light fuel oil, automotive diesel prices (USD), EU ETS carbon prices</td><td>IEA, World Bank</td></tr><tr><td>International Trade</td><td>Export/import indices (chemicals, machinery, transport equipment, fuels, raw materials)</td><td>Eurostat</td></tr></table>

Table 1: Input variables $\boldsymbol { x } \in \mathbb { R } ^ { d }$ used in our study, and their sources.
<table><tr><td>Sector</td><td>Activity according to Eurostat definitions</td></tr><tr><td>HeatingCooling</td><td>Heating and cooling by households</td></tr><tr><td>Industry</td><td>Manufacturing, mining, construction</td></tr><tr><td>Land</td><td>Agriculture, forestry, fishing, water supply, waste management</td></tr><tr><td>Mobility</td><td>Transportation, storage, trade, vehicle repair, transport by households</td></tr><tr><td>Power</td><td>Electricity, gas, steam, air conditioning</td></tr><tr><td>Other</td><td>Health, science, technical and professional services, finance, insurance, information, communication, public admin., arts, entertainment, defense, administrative services, real estate, accommodation, food service, education, other services, and household activities</td></tr></table>

Table 2: Target emission variables $\mathbf { y } \in \mathbb { R } ^ { s }$ , gathered from Eurostat.

Forecasting is performed in the latent space rather than the original indicator space, as the VAE’s structured low-dimensional representation makes year-to-year transitions smoother and more regular than the raw indicators. The latent forecaster $f _ { \theta }$ predicts:

$$
\hat { \mathbf { z } } ( t , i ) = f _ { \theta } ( \mathbf { z } ( t - 1 , i ) , \mathbf { z } ( t - 2 , i ) , \mathbf { c } ( t , i ) , \mathbf { c } ( t - 1 , i ) )
$$

iterated from the last observed year through 2030. Context variables $\mathbf { c } ( t , i )$ , comprising GDP, population, and climate variables, enter the forecaster directly rather than through the encoder, as their near-term trajectories are comparatively tractable to project and their influence on emissions may operate through mechanisms not fully captured by the latent representation. The emission predictor $g _ { \psi }$ then maps the projected latent sequence and context variables to sectoral outputs:

$$
\hat { \mathbf { y } } ( t , i ) = g _ { \psi } ( \hat { \mathbf { z } } ( t , i ) , \hat { \mathbf { z } } ( t - 1 , i ) , \mathbf { c } ( t , i ) , \mathbf { c } ( t - 1 , i ) )
$$

alongside a learned uncertainty vector u $( t , i ) \in \mathbb { R } ^ { s }$ , detailed in Section 4.3.2.

This framework does not embed policy targets, assume future technology adoption, or model behavioral responses to price signals. It learns emission dynamics as they have manifested historically and extrapolates those dynamics forward at their observed pace

## 4.3 Model Architecture

This section details our uncertainty-aware framework for modeling and interpreting macroeconomic drivers of sectoral carbon emissions. We begin by describing our variational autoencoder architecture that captures meaningful country-year representations, followed by our emission prediction model that leverages these latents, to provide both accurate predictions and uncertainty estimates, then we detail the model forecasting the latent features into the future.

## 4.3.1 Variational Autoencoder

The VAE maps transition indicators $\mathbf { x } ( t , i )$ to a probabilistic latent state $\mathbf { z } ( t , i ) \sim N ( \mu _ { \phi } ( \mathbf { x } ) , \sigma _ { \phi } ^ { 2 } ( \mathbf { x } ) )$ [17].

Encoder Network The encoder network maps input variables $\boldsymbol { x } \in \mathbb { R } ^ { d }$ to parameters of a latent distribution $z \sim \mathcal { N } ( \mu , \sigma ^ { 2 } )$ where $\mu ,$ log $\sigma ^ { 2 } \in \mathbb { R } ^ { k }$ . We implement the encoder as a series of residual blocks connected by linear layers:

$$
h _ { \mathrm { e n c } } ( x ) = ( \mu ( x ) , \log \sigma ^ { 2 } ( x ) )
$$

Each residual block contains the following components:

• A skip connection[10] to facilitate gradient flow during training. These prove useful for stable training and fast convergence.

• Linear transformations with GELU[11] activations, which empirically outperformed other nonlinearities.

• Dropout[36] regularization to prevent overfitting, applied both to neuron activations and to input features.

• No normalization layers, as we found empirically that they decreased model performance.

Decoder Network The decoder mirrors the encoder’s architecture but reverses the information flow, mapping samples from the latent distribution back to the original input space:

$$
h _ { \mathrm { d e c } } ( z ) = \hat { x }
$$

We sample from the latent distribution using the reparameterization trick to enable backpropagation through the sampling process:

$$
z = \mu ( x ) + \sigma ( x ) \odot \epsilon , \quad \epsilon \sim N ( 0 , I )
$$

VAE Loss Function We train the VAE using a weighted combination of reconstruction loss and KL divergence regularization:

$$
\mathcal { L } _ { \mathrm { V A E } } = \lambda _ { L 1 } \mathcal { L } _ { \mathrm { r e c o n } } + \lambda _ { K L } \mathcal { L } _ { \mathrm { K L } }
$$

where $\lambda _ { L 1 } = 0 . 9 5$ and $\lambda _ { K L } = 0 . 0 5$ are weight hyperparameters. For the reconstruction loss $\scriptstyle { \mathcal { L } } _ { \mathrm { r e c o n } }$ , we employ the L1 norm (Mean Absolute Error): $\mathcal { L } _ { \mathrm { r e c o n } } = \| x - \hat { x } \| _ { 1 }$ <sub>1</sub>. Empirically, we found that L1 yielded better convergence and more meaningful latent representations than MSE for our heterogeneous dataset. The KL divergence term $\mathcal { L } _ { \mathrm { K I } }$ regularizes the latent distribution towards a standard Gaussian:

$$
\mathcal { L } _ { \mathrm { K L } } = D _ { \mathrm { K L } } ( N ( \mu ( x ) , \sigma ^ { 2 } ( x ) ) \| N ( 0 , I ) )
$$

Implementation Details The VAE was trained for 5000 epochs using AdamW[20] with a learning rate of $5 \times 1 0 ^ { - 4 } , \epsilon = 1 0 ^ { - 6 }$ , and a high weight decay of 0.1. To ensure stable training, we applied gradient norm clipping to 1.0[47]. We use 0.33 dropout in every layer and 0.4 input dropout. We use 5 residual blocks, with 6 hidden layers each for the encoder, with a number of neurons linearly decreasing in each block, from the input dimension to the latent size. Hyperparameters were found following a Bayesian optimization. Using CodeCarbon[5], we measure our electricity consumption: training the VAE uses 5.46 Wh of electricity, which amount to emitting 1.81 $\mathrm { \ g } \mathrm { C O } _ { 2 } \mathrm { e q }$ at the carbon intensity of Italy’s national electricity grid.

## 4.3.2 Emissions Prediction Model

The emission predictor $g _ { \psi }$ maps projected latent states and context variables to sectoral emission estimates, and is trained after the VAE on its fixed latent representations. We pair $g _ { \psi }$ with an uncertainty-aware loss [16] that trains it to jointly estimate sectoral outputs and a per-sector confidence metric, penalizing large residuals while permitting higher uncertainty where emissions are intrinsically more variable or dificult to extrapolate.

Temporal Context Integration To capture temporal dynamics in emissions patterns, our predictor incorporates both current and historical information. For each country-year data point, we sample two latent representations:

1. The current year’s latent state $z _ { t } \sim N ( \mu ( x _ { t } ) , \sigma ^ { 2 } ( x _ { t } ) )$

2. The previous year’s latent state $z _ { t - 1 } \sim N ( \mu ( x _ { t - 1 } ) , \sigma ^ { 2 } ( x _ { t - 1 } ) )$

For cases where previous year data is unavailable (e.g., the first year in our dataset for each country), we use zero vectors as placeholders. Additionally, we include context variables from both the current and previous year explicitely, since these may impact emissions through mechanisms not fully captured by the latent representation. This approach allows the model to identify both absolute states and year-to-year changes that afect emissions.

The combined input to the predictor network is:

$$
\nu = \left[ z _ { t } ; z _ { t - 1 } ; c _ { t } ; c _ { t - 1 } \right]
$$

where $c _ { t }$ and $c _ { t - 1 }$ represent climate variables for the current and previous years, respectively, and [; ] denotes vector concatenation.

Dual Output Architecture Our predictor returns two outputs:

• Emissions predictions $\hat { y } \in \mathbb { R } ^ { 2 s }$ , consisting ofper capita emissions across $s = 6$ sectors

• Uncertainty estimates $\sigma _ { \boldsymbol { y } } \in \mathbb { R } _ { + } ^ { s }$ , representing the model’s confidence in its predictions for each sector

This architecture follows a similar design to the VAE, employing residual blocks with GELU activations and dropout regularization. We use a network with wider hidden layers compared to the VAE, as experiments indicated this improved predictive performance.

Uncertainty-Aware Loss Function We train the emissions predictor using an uncertainty-aware loss function[16] that balances prediction accuracy with uncertainty calibration. The loss function combines mean squared error with learned uncertainty, following the principled approach proposed in uncertainty quantification literature:

$$
\mathcal { L } = \frac { 1 } { 2 } \exp ( - \log \sigma _ { y } ^ { 2 } ) \cdot ( y - \hat { y } ) ^ { 2 } + \frac { 1 } { 2 } \log \sigma _ { y } ^ { 2 }
$$

This formulation has two key properties. First, it automatically balances the error term with the uncertainty regularization. When prediction errors are high, the model is incentivized to increase its uncertainty estimates. Conversely, log $\sigma _ { y } ^ { 2 }$ prevents the model from assigning arbitrarily high uncertainty to avoid the error penalty. By jointly optimizing for total and per capita emissions with shared uncertainty estimates, the model can develop a more robust representation of emission drivers while appropriately quantifying uncertainty.

Implementation Details We trained the predictor for 5000 epochs using AdamW [20] with a learning rate of $3 \times 1 0 ^ { - 5 } , \epsilon = 1 0 ^ { - 6 }$ , and high weight decay of 0.22. As with the VAE, we applied gradient norm clipping to 1.0 to ensure stable training. We use dropout of 0.09. We use 2 residual blocks with 2 hidden layers each, with 128 neurons each and SILU activation [7]. We train all our models using $\mathrm { P y } ^ { \prime }$ Torch[31]. Hyperparameters were found following a Bayesian optimization with WandB[3]. Using CodeCarbon[5], we measure our electricity consumption: training the predictor uses 4.61 Wh of electricity, which amount to emitting 1.52 g CO<sub>2</sub>eq at the carbon intensity of Italy’s national electricity grid.

## 4.3.3 Latent-Space Forecasting Model

The latent forecaster $f _ { \theta }$ learns the temporal evolution of the VAE latent space, enabling autoregressive projections that preserve the cross-indicator covariance structure encoded by $q _ { \phi }$

Temporal Context Integration To model temporal dependencies, the forecasting network leverages information from multiple previous years. For each country-year couple, we construct an input vector that combines:

1. A sample of the latent state of the previous year: $z _ { t - 1 } = \mu ( x _ { t - 1 } )$

2. A sample of the latent state two years prior: $z _ { t - 2 } = \mu ( x _ { t - 2 } )$

3. Climate and macroeconomic context variables for the current year $c _ { t }$

4. The same context variables from the previous year $c _ { t - 1 }$

The concatenated input of the forecast model is:

$$
\boldsymbol { u } = \left[ z _ { t - 1 } ; z _ { t - 2 } ; c _ { t } ; c _ { t - 1 } \right]
$$

Forecast Network Architecture The forecasting model $f _ { \theta }$ maps $u _ { t }$ to a prediction of the next latent representation:

$$
\hat { z _ { t } } = f _ { \theta } ( u _ { t } )
$$

We implement $f _ { \theta }$ as a compact feed-forward network employing residual [10] blocks, GELU [11] activations, and dropout [36] regularization—mirroring the architectural principles used in the emissions predictor. Residual connections facilitate stable training by allowing gradients to propagate through the forecast model efectively. The network predicts a vector in the latent space dimension, ensuring compatibility with both the VAE encoder and the emissions predictor.

Learning Samples The predicted latent state $z ^ { t }$ aims to be a direct sample from the target latent distribution. We perform this by training using the uncertainty-aware loss function[16] balancing closeness with the target mean $\mu ( x _ { t } )$ , and calibrating with the target variance $\sigma ^ { 2 } ( x _ { t } )$

For each latent dimension, we compute the loss as:

$$
\mathcal { L } = \frac { 1 } { 2 } \exp ( - \log \sigma _ { y } ^ { 2 } ( x _ { t } ) ) \cdot | \mu _ { y } ( x _ { t } ) - \hat { z } _ { y , t } | + \frac { 1 } { 2 } \log \sigma ^ { 2 } ( x _ { t } )
$$

This loss stabilizes training and encourages the forecaster to respect the probabilistic structure encoded by the ${ \mathrm { V A E } } ,$ rather than overfitting deterministic latent trajectories.

Implementation Details We trained the forecaster for 5000 epochs using AdamW [20] with a learning rate of $5 \times 1 0 ^ { - 4 } , \epsilon = 1 0 ^ { - 6 }$ , and high weight decay of 0.09. As with the other models, we applied gradient norm clipping to 1.0 to ensure stable training. We use dropout of 0.15. We use 3 residual blocks with 5 hidden layers each, with 128 neurons each and GELU activation [11]. We train all our models using PyTorch[31]. Hyperparameters were found following a Bayesian optimization with WandB[3]. Using CodeCarbon[5], we measure our electricity consumption: training the forecaster uses 29.86 Wh of electricity, which amount to emitting 9.87 g CO<sub>2</sub>eq at the carbon intensity of Italy’s national electricity grid.

## 4.4 Projection Procedure

Emission projections from 2025 to 2030 are generated autoregressively for each EU27 member state. The latent forecasting chain is initialized from the 2022 and 2023 observed indicator data, which constitute the last two years of the training set. For each projection year, the latent forecaster $f _ { \theta }$ advances the latent state forward by one step, conditioned on projected context variables, and the emission predictor $g _ { \psi }$ then maps the resulting latent state to per-sector emission change relative to the previous year.

The 2024 projection year requires special treatment. While observed sectoral $\mathrm { C O } _ { 2 }$ emissions are available for 2024 from Eurostat, the broader indicator dataset used to train and drive the encoder is not yet suficiently complete to robustly condition the latent chain from that year forward. We therefore use the latent forecaster to obtain a 2024 latent representation, which serves as the initialisation point for the 2025–2030 projection chain. Emission deltas are accumulated from 2025 onward, anchored to the observed 2024 sectoral values. This ensures that projections are grounded in the most recent available emission data, despite the latent representation for that year being derived from the forecaster rather than directly encoded from indicators.

Uncertainty is introduced at the emission prediction step only: for each Monte Carlo sample, the encoder’s latent distribution $z ( t , i ) \sim \smash { N ( \mu ( x ) , \sigma ^ { 2 } ( x ) ) }$ is resampled at inference time, while the latent forecasting chain is held fixed at the posterior mean $\mu ( x )$ . This captures uncertainty in the mapping from observed indicators to the latent state and onward to sectoral emissions, but does not encompass uncertainty in the latent dynamics themselves. The resulting intervals therefore reflect uncertainty in the emission mapping given the expected development of each EU27 member state, rather than uncertainty over the trajectory itself.

## 4.5 Uncertainty Interval Calibration

The framework produces two distinct uncertainty representations. The learned output ${ \bf \delta u } ( t , i )$ of the emission predictor is a model-internal indicator of relative sectoral predictability. Separately, the Monte Carlo confidence intervals reported in Figure 3 are obtained by repeatedly resampling the encoder’s latent distribution when predicting emissions, as described below.

The Monte Carlo projections sample the encoder’s latent distribution $z _ { t } \sim N ( \mu ( x _ { t } ) , \sigma ^ { 2 } ( x _ { t } ) )$ repeatedly at the emission prediction step, with the latent forecasting chain fixed at the posterior mean $\mu ( x _ { t } )$ , generating a spread over emission outcomes that reflects uncertainty in the mapping from observed indicators to the latent state and onward to sectoral emissions. This source of uncertainty is partial: it does not encompass uncertainty in the latent forecasting step, the predictor’s learned sector-level confidence, or structural sources such as future shocks or measurement error in the input indicators. The intervals are therefore not predictive distributions over future outcomes but model-consistency bands, and are most informative for comparing relative predictability across sectors and countries.

Within this scope, the raw encoder spread need not be consistent with the model’s empirical prediction errors. Following [18], we apply a post-hoc rescaling to ensure that the reported bands achieve 90% coverage on the historical period 2010–2023. For each country-sector cell $( i , s )$ we draw $N = 5 0 0$ emission samples per historical observation by re-sampling $z _ { t }$ and passing each sample through the predictor, yielding an empirical mean $\hat { y } _ { i , s , t }$ and standard deviation $\hat { \sigma } _ { i , s , t }$ . The standardised residual $\boldsymbol { r } _ { i , s , t } = \left( y _ { i , s , t } - \hat { y } _ { i , s , t } \right) / \hat { \sigma } _ { i , s , t }$ normalises prediction errors by the model’s estimated spread, allowing direct comparison against a standard normal distribution. We estimate the temperature scalar:

$$
T _ { i , s } = \frac { q u a n t i l e _ { 0 . 9 0 } ( | r _ { i , s , t } | ) } { 1 . 6 4 5 }
$$

and rescale each MC sample as $y _ { \mathrm { c a l } , k } = \bar { y } + T _ { i , s } ( y _ { k } - \bar { y } )$ , leaving the mean unchanged while adjusting the spread to match the empirical residual distribution. $T _ { i , s } > 1$ indicates that the raw encoder spread underestimates historical errors; $T _ { i , s } < 1$ indicates over-dispersion. Aggregate intervals sum the calibrated sample vectors across sectors before taking percentiles, preserving the joint distribution rather than summing marginal quantiles independently.

## 4.6 Limitations

Several limitations of the present framework should be kept in mind when interpreting our projections.

First, our framework models $\mathrm { C O } _ { 2 }$ emissions only, excluding non- $\mathbf { \cdot C O } _ { 2 }$ gases such as methane and nitrous oxide whose sectoral dynamics are more erratic and governed by distinct processes. Comparisons against the FF55 target, which formally applies to all greenhouse gases, should therefore be interpreted as indicative rather than exact.

Second, the indicator panel underlying our projections is observed only through 2023, with 2024 reconstructed via the forecaster rather than directly encoded (Section 4.4). This lag means the mode cannot reflect the most recent shifts in policy, prices, or technology adoption. As new annual data become available, the framework can be re-estimated to narrow this gap, but, at any point in time, the model’s most recent panel will lag the present by the publication delay of its training data sources.

Third, by design, the framework extrapolates empirical momentum without conditioning on future policy interventions, behavioral responses, or structural shocks. It will therefore systematically miss two classes of discontinuities: exogenous shocks of the kind documented in the rolling-origin evaluation, such as the COVID-19 pandemic, and abrupt changes in the pace or ambition of national climate policy that have no precedent in the historical record. The resulting projections are best interpreted as a current-trends counterfactual rather than a forecast of realized outcomes.

Fourth, while we make a substantial efort to account for uncertainty, producing both learned sectorlevel confidence scores and calibrated Monte Carlo consistency bands, we acknowledge that these fall short of formal prediction intervals in the classical statistical sense. Model confidence scores are best interpreted as internal comparative measures of relative sectoral predictability, and are not safely translatable into probabilistic statements. The consistency bands, while calibrated against historical model errors, cannot be confidently interpreted as covering the true range of future outcomes, and should not be treated as predictive distributions over the projection horizon.

Finally, while the gradient-based attribution and sensitivity analyses provide high-level insight into the model’s internal structure, they do not constitute a causal inference analysis. The associations identified between transition indicators and sectoral emissions reflect patterns learned from historical co-movement rather than structural causal relationships, and, while useful for validating the model’s mechanisms and gathering high-level insights into the drivers of emission dynamics, should not be interpreted as explicit policy levers or quantitative estimates of the efect of specific interventions on emission trajectories.

## Acknowledgments

J. Ghirri, C. Rodriguez-Pardo, and M. Tavoni acknowledge funding from the European Union European Research Council (ERC), Grant project No 101044703 (EUNICE).

## References

[1] Moloud Abdar, Farhad Pourpanah, Sadiq Hussain, Dana Rezazadegan, Li Liu, Mohammad Ghavamzadeh, Paul Fieguth, Xiaochun Cao, Abbas Khosravi, U Rajendra Acharya, and oth-

ers. A review of uncertainty quantification in deep learning: Techniques, applications and challenges. Information Fusion, 76:243–297, 2021.

[2] Christoph Bertram, Elina Brutschin, Laurent Drouet, Gunnar Luderer, Bas van Ruijven, Lara Aleluia Reis, Luiz Bernardo Baptista, Harmen-Sytze de Boer, Ryna Cui, Vassilis Daioglou, Florian Fosse, Dimitris Fragkiadakis, Oliver Fricko, Shinichiro Fujimori, Nate Hultman, Gokul Iyer, Kimon Keramidas, Volker Krey, Elmar Kriegler, Robin D. Lamboll, Rahel Mandaroux, Pedro Rochedo, Joeri Rogelj, Roberto Schaefer, Diego Silva, Isabela Tagomori, Detlef van Vuuren, Zoi Vrontisi, and Keywan Riahi. Feasibility of peak temperature targets in light of institutional constraints. Nature Climate Change, 14(9):954–960, September 2024.

[3] Lukas Biewald. Experiment Tracking with Weights and Biases, 2020.

[4] Elina Brutschin, Silvia Pianta, Massimo Tavoni, Keywan Riahi, Valentina Bosetti, Giacomo Marangoni, and Bas J van Ruijven. A multidimensional feasibility evaluation of low-carbon scenarios. Environmental Research Letters, 16(6):064069, June 2021.

[5] Benoit Courty, Victor Schmidt, Sasha Luccioni, Goyal-Kamal, MarionCoutarel, Boris Feld, Jérémy Lecourt, LiamConnell, Amine Saboni, Inimaz, supatomic, Mathilde Léval, Luis Blanche, Alexis Cruveiller, ouminasara, Franklin Zhao, Aditya Joshi, Alexis Bogrof, Hugues de Lavoreille, Niko Laskaris, Edoardo Abati, Douglas Blank, Ziyao Wang, Armin Catovic, Marc Alencon, Michał Stęchły, Christian Bauer, Lucas Otávio N. de Araújo, JPW, and MinervaBooks. mlco2/codecarbon: v2.4.1, May 2024.

[6] Yirui Deng, Mengjuan Yin, Xiaofeng Xu, Lean Yu, Guowei Gao, and Li Ma. How to develop global energy-intensive sectors in the presence of carbon tarifs? Journal of International Financial Markets, Institutions and Money, 91:101930, March 2024.

[7] Stefan Elfwing, Eiji Uchibe, and Kenji Doya. Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning, November 2017. arXiv:1702.03118 [cs].

[8] Shuo Feng, Yu Gu, Yanyan Zhao, Lei Chen, Xu Tang, Xinkai Qiao, and Danli Wang. EVCBiLNet: Carbon Emission forecasting Model Based on CNN-BiLSTM and Secondary Feature Decomposition. Emission Control Science and Technology, 11(2):28, December 2025.

[9] Karl Pearson F.R.S. LIII. On lines and planes of closest fit to systems of points in space. The London, Edinburgh, and Dublin Philosophical Magazine and Journal ofScience, 2(11):559–572, 1901.

[10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.

[11] Dan Hendrycks and Kevin Gimpel. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415, 2016.

[12] Intergovernmental Panel On Climate Change (Ipcc). Climate Change 2021 – The Physical Science Basis: Working Group I Contribution to the Sixth Assessment Report ofthe Intergovernmental Panel on Climate Change. Cambridge University Press, 1 edition, July 2023.

[13] Jessica Jewell and Aleh Cherp. On the political feasibility of climate change mitigation pathways: Is it too late to keep warming below 1.5°C? WIREs Climate Change, 11(1):e621, 2020. \_eprint: https://wires.onlinelibrary.wiley.com/doi/pdf/10.1002/wcc.621.

[14] Jessica Jewell and Aleh Cherp. The feasibility of climate action: Bridging the inside and the outside view through feasibility spaces. WIREs Climate Change, 14(5):e838, 2023. \_eprint: https://wires.onlinelibrary.wiley.com/doi/pdf/10.1002/wcc.838.

[15] Yukai Jin, Ayyoob Sharifi, Zhisheng Li, Sirui Chen, Suzhen Zeng, and Shanlun Zhao. Carbon emission prediction models: A review. Science ofThe Total Environment, 927:172319, June 2024.

[16] Alex Kendall and Yarin Gal. What uncertainties do we need in bayesian deep learning for computer vision? Advances in neural information processing systems, 30, 2017.

[17] Diederik P. Kingma and Max Welling. Auto-Encoding Variational Bayes, December 2022. arXiv:1312.6114 [stat].

[18] Volodymyr Kuleshov, Nathan Fenner, and Stefano Ermon. Accurate Uncertainties for Deep Learning Using Calibrated Regression, July 2018. arXiv:1807.00263 [cs].

[19] Feifei Li, Zhe Xu, and Hui Ma. Can China achieve its CO2 emissions peak by 2030? Ecological Indicators, 84:337–344, January 2018.

[20] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017.

[21] Stefan L. Luxembourg, Steven S. Salim, Koen Smekens, Francesco Dalla Longa, and Bob van der Zwaan. TIMES-Europe: An Integrated Energy System Model for Analyzing Europe’s Energy and Climate Challenges. Environmental Modeling & Assessment, 30(1):1–19, February 2025.

[22] Aysha Malik, Ejaz Hussain, Sofia Baig, and Muhammad Fahim Khokhar. Forecasting CO2 emissions from energy consumption in Pakistan under diferent scenarios: The China–Pakistan Economic Corridor. Greenhouse Gases: Science and Technology, 10(2):380–389, 2020. \_eprint: https://scijournals.onlinelibrary.wiley.com/doi/pdf/10.1002/ghg.1968.

[23] Michael D. Mastrandrea, Katharine J. Mach, Gian-Kasper Plattner, Ottmar Edenhofer, Thomas F. Stocker, Christopher B. Field, Kristie L. Ebi, and Patrick R. Matschoss. The IPCC AR5 guidance note on consistent treatment of uncertainties: a common approach across the working groups. Climatic Change, 108(4):675, August 2011.

[24] Leland McInnes, John Healy, Nathaniel Saul, and Lukas Grossberger. UMAP: Uniform Manifold Approximation and Projection. The Journal ofOpen Source Software, 3(29):861, 2018.

[25] Malte Meinshausen, Jared Lewis, Christophe McGlade, Johannes Gütschow, Zebedee Nicholls, Rebecca Burdon, Laura Cozzi, and Bernd Hackmann. Realization of Paris Agreement pledges may limit warming just below 2 °C. Nature, 604(7905):304–309, April 2022.

[26] Seyed Mahdi Miraftabzdeh, Mohammed Ali Khan, Navid Bayati, and Dario Zaninelli. Deep learning methods and evaluation of the extensive carbon emission predictive solution for Danish grid. Sustainable Energy Technologies and Assessments, 75:104242, March 2025.

[27] Mihai Mutascu. CO2 emissions in the USA: new insights based on ANN approach. Environmental Science and Pollution Research, 29(45):68332–68356, 2022.

[28] Ahmed M Nassef, Abdul Ghani Olabi, Hegazy Rezk, and Mohammad Ali Abdelkareem. Application of artificial intelligence to predict CO2 emissions: critical step towards sustainable environment. Sustainability, 15(9):7648, 2023.

[29] Naomi Oreskes, David A Stainforth, and Leonard A Smith. Climate change prediction: Erring on the side of least drama? Global Environmental Change, 20(1):1–2, 2010.

[30] Hsiao-Tien Pao, Hsin-Chia Fu, and Cheng-Lung Tseng. Forecasting of CO2 emissions, energy consumption and economic growth in China using an improved grey model. Energy, 40(1):400–409, April 2012.

[31] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. PyTorch: an imperative style, high-performance deep learning library. In Proceedings of the 33rd International Conference on Neural Information Processing Systems. Curran Associates Inc., Red Hook, NY, USA, 2019.

[32] Keywan Riahi, Christoph Bertram, Daniel Huppmann, Joeri Rogelj, Valentina Bosetti, Anique-Marie Cabardos, Andre Deppermann, Laurent Drouet, Stefan Frank, Oliver Fricko, Shinichiro Fujimori, Mathijs Harmsen, Tomoko Hasegawa, Volker Krey, Gunnar Luderer, Leonidas Paroussos, Roberto Schaefer, Matthias Weitzel, Bob van der Zwaan, Zoi Vrontisi, Francesco Dalla Longa, Jacques Després, Florian Fosse, Kostas Fragkiadakis, Mykola Gusti, Florian Humpenöder, Kimon Keramidas,

Paul Kishimoto, Elmar Kriegler, Malte Meinshausen, Larissa P. Nogueira, Ken Oshiro, Alexander Popp, Pedro R. R. Rochedo, Gamze Ünlü, Bas van Ruijven, Junya Takakura, Massimo Tavoni, Detlef van Vuuren, and Behnam Zakeri. Cost and attainability of meeting stringent climate targets without overshoot. Nature Climate Change, 11(12):1063–1069, December 2021.

[33] Keywan Riahi, Detlef P Van Vuuren, Elmar Kriegler, Jae Edmonds, Brian C O’neill, Shinichiro Fujimori, Nico Bauer, Katherine Calvin, Rob Dellink, Oliver Fricko, and others. The shared socioeconomic pathways and their energy, land use, and greenhouse gas emissions implications: an overview. Global Environmental Change, 42:153–168, 2017.

[34] Andrea Saltelli, Paola Annoni, Ivano Azzini, Francesca Campolongo, Marco Ratto, and Stefano Tarantola. Variance based sensitivity analysis of model output. Design and estimator for the total sensitivity index. Computer Physics Communications, 181(2):259–270, February 2010.

[35] I. M. Sobol’. Global sensitivity indices for nonlinear mathematical models and their Monte Carlo estimates. Mathematics and Computers in Simulation, 2001.

[36] Nitish Srivastava, Geofrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. Dropout: a simple way to prevent neural networks from overfitting. The journal of machine learning research, 15(1):1929–1958, 2014.

[37] Laurens Van der Maaten and Geofrey Hinton. Visualizing data using t-SNE. Journal ofmachine learning research, 9(11), 2008.

[38] Kawalpreet Vyas and Zakir Ahmed. Impact of economic growth, energy consumption and population on carbon emissions: Evidence from India. Indian Journal of Economics and Development, 1(3):90–96, 2013.

[39] Rupert Way, Matthew C. Ives, Penny Mealy, and J. Doyne Farmer. Empirically grounded technology forecasts and the energy transition. Joule, 6(9):2057–2082, September 2022.

[40] Lei Wen and Yang Cao. A hybrid intelligent predicting model for exploring household CO2 emissions mitigation strategies derived from butterfly optimization algorithm. Science ofThe Total Environment, 727:138572, July 2020.

[41] Wei Wu, Hongwei Zheng, and Xiaolu Guan. Application of artificial neural network in the estimation of carbon dioxide emissions. Journal ofCleaner Production, 199:715–729, 2018.

[42] Huijuan Yang and John F. O’Connell. Short-term carbon emissions forecast for aviation industry in Shanghai. Journal ofCleaner Production, 275:122734, December 2020.

[43] Yuan Yang, Junjie Zhang, and Can Wang. Forecasting China’s Carbon Intensity: Is China on Track to Comply with Its Copenhagen Commitment? The Energy Journal, 39(2):63–86, March 2018.

[44] Liang Ye, Pei Du, and Shubin Wang. Industrial carbon emission forecasting considering external factors based on linear and machine learning models. Journal of Cleaner Production, 434:140010, January 2024.

[45] Yang Yu, Yu-ru Deng, and Fei-fan Chen. Impact of population aging and industrial structure on CO2 emissions and emissions trend prediction in China. Atmospheric Pollution Research, 9(3):446–454, May 2018.

[46] Yang Yu and Wei Xu. Impact of FDI and R&D on China’s industrial CO2 emissions reduction and trend prediction. Atmospheric Pollution Research, 10(5):1627–1635, September 2019.

[47] Jingzhao Zhang, Tianxing He, Suvrit Sra, and Ali Jadbabaie. Why gradient clipping accelerates training: A theoretical justification for adaptivity. arXiv preprint arXiv:1905.11881, 2019.

## Supplementary

## Latent representation inspection

To assess the quality of the VAE latent representation, Figures 6 and 7 compare the structure of the original input space and the learned latent space using t-SNE [37] and UMAP [24] dimensionality reduction respectively.

![](images/2f868067ba480da2f68e5225d0e666e59af042e32643aa45dfb3cfd1b1240714.jpg)

![](images/2aab3ea9ebd23c6da3b5f3516a592829de0a46cf47e8c60ef59d2f8bde83a0e8.jpg)  
Figure 6: t-SNE projections of (a) the original high-dimensional input space and (b) the VAE latent space across all EU27 countries and years. Each point represents a country-year observation; connected points trace the temporal trajectory of each country. The latent space shows markedly improved separation between countries compared to the dense clustering observed in the original input space, indicating that the VAE learns a more structured and disentangled representation of emission-relevant characteristics.

In the original high dimensional input space, both visualization reveal a characteristic pattern: the largest economies separate as clear outliers, driven primarily by scale diferences, while most other member states collapse into a dense, poorly diferentiated central cluster. The VAE latent space substantially improves this structure. Country-level trajectories become more spatially separated and temporally coherent, with nations that share similar decarbonisation profiles, such as the Nordic countries or the coal-dependent Central European states, clustering more naturally. This suggests that the VAE learns to organise countries along transition-relevant structural dimensions rather than being dominated by raw economic scale. Nevertheless, the latent trajectories are not perfectly smooth: abrupt jumps remain visible for several countries, reflecting genuine structural discontinuities in the underlying data, most notably the COVID-19 demand shock in 2020 and the energy price disruptions following the onset of the war in Ukraine in 2022, both of which produced sharp, simultaneous shifts across multiple input indicators. These discontinuities are not artifacts of the compression but rather faithful reflections of real regime disruptions that the VAE appropriately preserves rather than smooths away. Overall, both t-SNE and UMAP confirm that the low dimensional latent space retains meaningful neighborhood structure while achieving a more disentangled and interpretable organization of the country-year observations.

## Model Performance

Performance metrics for all three model components are reported in Table 3, computed on held-out validation sets corresponding to a 15% random split of the training data, using the same split as model selection during training. All predictor metrics are reported in the z-score normalized emission space used during training (normalized kg $\mathrm { C O } _ { 2 }$ per capita); the latent forecaster metrics are in the normalized latent space. These units are not directly comparable to physical $\mathrm { C O } _ { 2 }$ quantities but provide a consistent basis for assessing relative fit across sectors and model components.

The VAE achieves stable reconstruction with near-identical KL divergence across train and validation splits (0.819 vs 0.818), indicating that regularization toward the standard Gaussian prior is consistent and does not depend on the specific observations seen during training. The small gap between training and validation reconstruction loss (0.124 vs 0.179) is consistent with expectations for a heterogeneous panel dataset where some country-year combinations are inherently harder to reconstruct.

![](images/c3398a51a45f328fcf6fd38aed7c75fe6f97421a6f957978f6ec93ac36e11c35.jpg)

![](images/dfd1248bcf7f20cd9e2de9eb7b3c6f6d47b22652cf47ad3785e8ca7ff060b3ce.jpg)  
Figure 7: UMAP projections of (a) the original high-dimensional input space and (b) the VAE latent space across all EU27 countries and years. Each point represents a country-year observation; connected points trace the temporal trajectory of each country. Compared to the original space, the latent representation exhibits greater country separation and more continuous temporal trajectories, reflecting the VAE’s ability to organise countries along meaningful structural dimensions while preserving global neighbourhood relationships.

The emission predictor achieves an aggregate $\mathbb { R } ^ { 2 }$ of 0.981 on the validation set, with per-sector values ranging from 0.964 (Power) to 0.991 (Land). Notably, validation metrics are slightly better than training metrics across all sectors (aggregate $\mathbb { R } ^ { 2 }$ of 0.969 on train vs 0.981 on val), which we attribute to the heavy regularization applied during training. High dropout rates create a harder efective training problem than the validation evaluation, which is conducted with dropout disabled. This pattern provides reassurance against overfitting. The relatively lower $\mathsf { R } ^ { 2 }$ for Power and Other on the validation set is consistent with the higher uncertainty estimates learned by the predictor for these sectors, as reported in Section 4.2 of the main text.

The latent forecaster shows minimal generalization gap, with train and validation MSE of 0.0062 and 0.0065 respectively. This near-identical performance across splits reflects the smoothness and regularity of latent trajectories learned by the VAE: because the encoder compresses heterogeneous inputs into a low-dimensional structured space, year-to-year transitions in this space are predictable and consistent across countries. We note that the random validation split used here does not constitute a temporal or spatial out-of-sample test for the forecaster.

## Rolling-Origin evaluation of the Forecasting Procedure

To assess whether the pipeline generalizes to genuinely unseen time periods rather than interpolating within the training window, we conduct a rolling-origin evaluation in which all three model components, VAE, emission predictor, and latent forecaster, are retrained from scratch on data up to a cutof year � and evaluated on autoregressive projections for years � + 1 through $T + 3$ . We repeat this procedure for $T \in \{ 2 0 1 8 , 2 0 1 9 , 2 0 2 0 , 2 0 2 1 \}$ }, generating emission projections for each held-out window and comparing against observed emissions from the full dataset. All metrics are reported in the z-score normalised emission space used during training

A critical interpretive note is required before reading these results. This evaluation compares model

Table 3: Model performance on held-out validation sets (15% random split, seed 0). Predictor metrics are reported in the z-score normalised emission space used during training (normalised kg CO per capita). Latent forecaster metrics are in the normalised latent space.

Panel A: Variational Autoencoder
<table><tr><td>Metric</td><td>Train</td><td>Validation</td></tr><tr><td>Reconstruction loss (L1)</td><td>0.124</td><td>0.179</td></tr><tr><td>KL divergence</td><td>0.819</td><td>0.818</td></tr><tr><td>Mean latent mean</td><td>0.006</td><td>0.013</td></tr><tr><td>Mean latent std</td><td>0.580</td><td>0.578</td></tr></table>

Panel B: Emission Predictor (validation split)
<table><tr><td>Metric</td><td>H&amp;C</td><td>Industry</td><td>Land</td><td>Mobility</td><td>Other</td><td>Power</td><td>Aggregate</td></tr><tr><td>MAE</td><td>0.070</td><td>0.095</td><td>0.062</td><td>0.083</td><td>0.103</td><td>0.086</td><td>0.083</td></tr><tr><td>RMSE</td><td>0.110</td><td>0.128</td><td>0.092</td><td>0.139</td><td>0.169</td><td>0.128</td><td>0.130</td></tr><tr><td>MSE</td><td>0.012</td><td>0.016</td><td>0.009</td><td>0.019</td><td>0.029</td><td>0.016</td><td>0.017</td></tr><tr><td>R²</td><td>0.985</td><td>0.972</td><td>0.991</td><td>0.988</td><td>0.969</td><td>0.964</td><td>0.981</td></tr></table>

Panel C: Latent Forecaster
<table><tr><td>Metric</td><td>Train</td><td>Validation</td></tr><tr><td>MSE</td><td>0.0062</td><td>0.0065</td></tr><tr><td>RMSE</td><td>0.0784</td><td>0.0805</td></tr><tr><td>MAE</td><td>0.0470</td><td>0.0486</td></tr></table>

projections against observed emissions in years where real policies were implemented, behavioral responses occurred, and structural shocks materialized, notably the COVID-19 demand collapse in 2020 and the energy price disruption following the onset of the war in Ukraine in 2022. By construction, our framework does not attempt to predict these events: it extrapolates empirical momentum assuming no further changes beyond natural evolution, producing a counterfactual trajectory rather than a forecast of what actually happened. Rolling-origin errors therefore reflect two distinct sources of divergence: genuine model error in extrapolating latent dynamics, and the gap between the counterfactual trajectory and a reality in which interventions and shocks did occur. These two components cannot be cleanly separated, but their joint presence means that the RMSE figures reported here should be interpreted as upper bounds on the model’s extrapolation error under stable conditions, rather than as estimates of forecast accuracy in the conventional sense.

Table 4 reports aggregate RMSE by training cutof and forecast horizon. Mean RMSE increases monotonically with horizon, from 0.227 at one year ahead to 0.326 at three years ahead, consistent with the expected accumulation of errors in an autoregressive system. The absolute magnitude of this degradation is modest: mean RMSE at horizon +3 is approximately 40% larger than at horizon +1, suggesting that the latent space learned by the VAE evolves suficiently smoothly that short-term extrapolation remains coherent across the three-year window relevant for our 2030 projections.

The variation across cutof years reflects the structural disruptions present in the evaluation periods rather than instability in the model itself. The 2018 cutof produces the highest RMSE at horizon +2 (0.416), corresponding to projections into 2020: the COVID-19 demand shock year. Observed emissions fell sharply across nearly all sectors and countries simultaneously, a pattern that no trend-extrapolation model trained on pre-pandemic data could anticipate, nor would be expected to. The 2019 and 2020 cutofs, whose evaluation windows partially overlap with the post-shock recovery period, show more moderate and stable performance, with RMSE values between 0.172 and 0.311 across horizons.

Table 4: Rolling-origin evaluation: aggregate RMSE (normalised emission space) for autoregressive projections of 1, 2, and 3 years ahead. Each row corresponds to models retrained from scratch on data up to the indicated cutof year.
<table><tr><td>Training cutoff</td><td>Horizon +1</td><td>Horizon +2</td><td>Horizon +3</td></tr><tr><td>2018</td><td>0.282</td><td>0.416</td><td>0.369</td></tr><tr><td>2019</td><td>0.261</td><td>0.225</td><td>0.263</td></tr><tr><td>2020</td><td>0.172</td><td>0.260</td><td>0.311</td></tr><tr><td>2021</td><td>0.191</td><td>0.279</td><td>0.363</td></tr><tr><td>Mean</td><td>0.227</td><td>0.295</td><td>0.326</td></tr></table>

Table 5 decomposes RMSE by sector and horizon, averaged across cutof years. The sectoral ordering is consistent with the uncertainty structure reported in the main text and with the nature of the evaluation periods. Land and Heating & Cooling show the lowest errors at all horizons (0.121–0.203 and 0.129–0.232 respectively), reflecting the structural regularity of these sectors and their relative insulation from the acute demand shocks that dominated the 2020–2022 period. Mobility and Power show the highest errors (0.253–0.471 and 0.305–0.345 respectively), which is expected: Mobility was among the sectors most severely disrupted by COVID-19 lockdowns, and Power underwent rapid structural change driven by both the renewable transition and the 2022 energy crisis, both of which represent genuine departures from pre-cutof trends rather than their continuation.

Table 5: Rolling-origin evaluation: RMSE by sector and forecast horizon, averaged across training cutofs � ∈ {2018, 2019, 2020, 2021}. Metrics are in the z-score normalised emission space used during training. Sectors are ordered by mean RMSE across horizons.
<table><tr><td>Sector</td><td>Horizon +1</td><td>Horizon +2</td><td>Horizon +3</td></tr><tr><td>Land</td><td>0.121</td><td>0.168</td><td>0.203</td></tr><tr><td>Heating &amp; Cooling</td><td>0.129</td><td>0.190</td><td>0.232</td></tr><tr><td>Industry</td><td>0.164</td><td>0.226</td><td>0.267</td></tr><tr><td>Mobility</td><td>0.253</td><td>0.366</td><td>0.471</td></tr><tr><td>Power</td><td>0.305</td><td>0.414</td><td>0.345</td></tr><tr><td>Other</td><td>0.325</td><td>0.363</td><td>0.379</td></tr></table>

Taken together, these results support the internal validity of the pipeline’s short-term projections under stable conditions. The monotonic horizon degradation, the concentration of elevated errors around known structural breaks, and the sectoral ordering of errors indicate that the model is learning economically coherent dynamics rather than overfitting to the training period. Where errors are large, they are large for interpretable reasons: they occur in the sectors and years where real-world dynamics departed most sharply from the kind of trend continuation that an inertial trajectory baseline embeds by construction.

## Model sensitivity

To characterize the model’s internal sensitivity structure and verify that predictions are anchored in economically coherent signals, we conduct a global variance-based sensitivity analysis using total-order Sobol indices [35, 34], reported in Figure 8. These results describe which inputs most strongly shape the model’s output variance; they are not estimates of real-world policy transmission. Given the aggressive dimensionality reduction preceding prediction, we expect the sensitivity structure to be highly concentrated, with most variables carrying negligible indices and predictive variance attributable to a small number of information-dense inputs.

![](images/61203cf80f83d18d09862778218afc12f573a7db18bda65ba21674f89567e30e.jpg)  
Figure 8: Sensitivity of sectoral CO emission predictions to input variables, measured by total-order Sobol indices (ST). Each cell reports the ST index for a given variable-sector pair. Variables are grouped thematically: fuel prices (left), macroeconomic indicators (centre), and electricity generation (right). The analysis is conducted on the full variable set, but only variables appearing in the top five for at least one sector are shown.

Fuel prices dominate the sensitivity structure across nearly all sectors. This is most naturally interpreted as a consequence of the dimensionality reduction, with fuel prices emerging as the most information-dense observable inputs, co-moving with the business cycle, energy market conditions, and geopolitical shocks simultaneously. This reflects the still strong dependence of national economies on fossil fuels. GDP shows concentrated sensitivity in Industry and Power, consistent with the income elasticity of energy-intensive production. Electricity generation variables carry uniformly low indices, likely because their co-movement with broader economic conditions is already absorbed into the latent representation.

## Sectoral emission changes, and model confidence, by country and sector

Figure 9 adds a model confidence dimension to the sectoral patterns described in Section 3.2. Power carries the highest confidence across member states, reflecting the structural regularity of the renewable transition and the predictability of its near-term trajectory. Mobility confidence is generally medium, including for several countries projecting near-zero changes, indicating genuine uncertainty about whether these tip marginally positive or negative, though the direction of the aggregate trend is robust. Heating & Cooling and Land Use carry the lowest confidence across most member states. Germany and Italy, identified in the main text as broad-based decarbonizers, maintain this profile at medium-to-high confidence across all six sectors.

![](images/94aa979c24977275d1f5f6069e7287daad976f2b414f0c561c5674dcc4ad1bfc.jpg)  
Figure 9: Projected sectoral $\mathrm { C O } _ { 2 }$ emission changes from 2024 to 2030 across EU27 member states. Each cell reports the percentage change in sector-level emissions between 2024 and 2030 under current trends. The bivariate color scheme encodes both the direction of change (blue: decreasing; red: increasing; grey: near zero, defined as within ±5%) and model confidence (light to dark, derived from the learned uncertainty of the emission predictor). The "Overall" row reports the aggregate change across all sectors. The EU27 column reports the emission-weighted aggregate across all member states. Bold separators distinguish the aggregate row and column from individual entries.