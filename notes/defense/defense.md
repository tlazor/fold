---
marp: true
theme: neobeam
paginate: true
math: katex
footer: '**Thom Lazor**
         **Measuring Linguistic Distance Using Spectral Overlap**'
---
<!-- _class: title -->
# Measuring Linguistic Distance Using Spectral Overlap
## Thom Lazor
## Advisor: Jan Šnajder
## University of Zagreb – FER
## July 2025
---

<!-- header: 'Table of contents' -->
**Motivation** 
Research questions & contributions 
Background  
Methodology  
Experiments  
Results  
Discussion & limitations  
Future work

---
<!-- header: 'Motivation' -->
Quantifying how *dissimilar* two languages are helps to  
* allocate teaching hours efficiently  
* guide model-transfer paths in low-resource NLP  
* reconstruct family trees in historical linguistics  
* prioritise endangered-language support

---
<!-- header: 'Gap in Existing Metrics' -->

* Producing distance matrices is **slow, uneven, or narrow in scope**.  
* Only *some* typological axes get measured, *some* languages get compared.  
* Often requires **costly, labour-intensive fieldwork**.

---
<!-- header: 'Opportunity: Language Models and Frequency Analysis' -->

* **Language models**  
  * Multilingual transformers internalise phonology, morphology, syntax, semantics.  
  * Every token yields **high-dimensional embedding / likelihood**

* **Text signals** : A sequence of embeddings or log-probs can be treated as a *time-series*

* **Frequency analysis looks past words**  
  * The Fourier transform highlights **periodicities** 
  * It ignores surface token identity and instead inspects **how** information is rhythmically distributed.

---
<!-- header: 'Table of contents' -->
Motivation
**Research questions & contributions**  
Background  
Methodology  
Experiments  
Results  
Discussion & limitations  
Future work

---
<!-- header: 'Research Questions' -->
1. **RQ1** – Do spectral metrics correlate with phonemic, lexical, cognitive & pedagogical baselines?  
2. **RQ2** – Are the metrics model- and dataset-agnostic?  
3. **RQ3** – Which frequency bands drive the correlation, and do they reflect typological traits?

---
<!-- header: 'Main Contributions' -->
* Introduce **Spectral KL** & **Spectral Coherence** distances  
* Evaluate 100 languages, 2 transformers, 2 corpora, 3 metrics  
* Map discriminative frequency bands to linguistic features

---

<!-- header: 'Table of contents' -->
Motivation
Research questions & contributions  
**Background**  
Methodology  
Experiments  
Results  
Discussion & limitations  
Future work

---

<!-- header: 'Background: Linguistic Subsystems' -->
| Subsystem | Captures | Typical distance |
|-----------|----------|------------------|
| Phonetics | Raw sounds | IPA edit distance |
| Phonology | Sound system rules | Jaccard inventories |
| Morphology | Word structure | Morphemes/word |
| Syntax | Phrase order | Tree-edit |
| Semantics | Meaning | Embedding cosine |

---
<!-- header: 'Baselines' -->
* **Phonemic** – ASJP LDND 
* **Lexical** – Swadesh Levenshtein  
* **Mutual intelligibility** – cloze tests  
* **FSI learning time** – weeks for English L1 diplomats

---
<!-- header: 'Multilingual Transformers' -->
* **mBERT** (110M params, 110k vocab, 104 langs)  
* **XLM-R** base (270M params, 250k vocab, 100 langs)  
* Shared sub-word vocabularies **→** aligned embedding space  

---
<!-- header: 'Fourier Transform' -->
x

---
<!-- header: 'Welchs Method' -->
x

---
<!-- header: 'Why Spectra?' -->
* Language encodes **periodicities** (stress, affixes, phrase lengths)  
* Comparing PSDs asks *“How is information rhythmically distributed?”*  
* Captures structure missed by token-level overlap

---
<!-- header: 'Periodicities in Text' -->
<style scoped>
  table.periodicities        { font-size: 0.72em; margin: 0 auto; }
</style>

<table class="periodicities">
  <thead>
    <tr>
      <th><strong>Device</strong></th>
      <th><strong>Scale</strong></th>
      <th><strong>Periodicity pattern</strong></th>
      <th><strong>Example</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Ablaut reduplication</td>
      <td>Token</td>
      <td>Fixed vowel alternation (e.g., <em>i–a–o</em>) inside paired morphemes</td>
      <td><em>ping-pong</em>, <em>tic-tac-toe</em></td>
    </tr>
    <tr>
      <td>Anadiplosis</td>
      <td>Word</td>
      <td>End-of-clause word becomes start of next clause</td>
      <td>Fear leads to <em>anger</em>; <em>anger</em> leads to <em>hate</em>; <em>hate</em> leads to suffering.</td>
    </tr>
    <tr>
      <td>Epistrophe</td>
      <td>Phrase</td>
      <td>Identical phrase closes successive units</td>
      <td>…of the <em>people</em>, by the <em>people</em>, for the <em>people</em>.</td>
    </tr>
    <tr>
      <td>Antithetical parallel</td>
      <td>Sentence</td>
      <td>Mirrored clauses with inverse relation</td>
      <td>Ask not what your country can do for you; ask what you can do for your country.</td>
    </tr>
  </tbody>
</table>

---
<!-- header: 'Coherence' -->

Coherence measures the degree to which one signal $x(t)$ predicts another $y(t)$ via an optimal least-squares estimator.

$
C_{xy}(f)=\frac{\lvert G_{xy}(f)\rvert^{2}}{G_{xx}(f)\,G_{yy}(f)}
$

$
G_{xy}(f) = \mathcal{F}\!\bigl\{R_{xy}(\tau)\bigr\}
          = \int_{-\infty}^{\infty} R_{xy}(\tau)\,e^{-j2\pi f\tau}\,d\tau
          = \mathbb{E}\!\bigl\{X(f)\,Y^{*}(f)\bigr\}
$

$R_{xy}(\tau)=\mathbb{E}\!\{x(t)\,y^{*}(t+\tau)\}$ is the cross-correlation function

---
<!-- header: 'Coherence' --> 

* **Geometric view:** $|\cos\theta|$ between time-series vectors 
* **Range:** $0 \le C_{xy}(f) \le 1$  
  * **0 → low coherence:** non-linear link, extra inputs, or high noise  
  * **1 → high coherence:** strong linear predictability  
* **Symmetric:** $|G_{xy}(f)| = |G_{yx}(f)|$
* **Practical use:** frequency-resolved health check  
  * High-$C_{xy}$ band ⇒ system behaving as modelled  
  * Low-$C_{xy}$ band ⇒ investigate fault (e.g., lubrication, impurities)

---
<!-- header: 'KL Divergence' -->

$
    D_{\mathrm{KL}}\!\bigl(P \,\Vert\, Q\bigr)
      \;=\;
      \sum_{i} P(i)\,\log\!\frac{P(i)}{Q(i)}
$

* **Interpretation:** the surprisal from using $Q$ as a model, when the true probability distribution is $P$

* **Non-negative:** $D_{\mathrm{KL}}\!\ge 0$  
* **Asymmetric:** $D_{\mathrm{KL}}(P \,\Vert\, Q)\neq D_{\mathrm{KL}}(Q \,\Vert\, P)$

---
<!-- header: 'Table of contents' -->
Motivation
Research questions & contributions  
Background 
**Methodology**   
Experiments  
Results  
Discussion & limitations  
Future work

---
<!-- header: 'Proposed Metrics' -->
- **Spectral Overlap**  
    $$
        PSO(l_1,l_2)=∑_f min(P_1(f), P_2(f))
    $$
- **Spectral KL Divergence**
    $$
        D_{\mathrm{KL}}\!\bigl(l_1 \,\Vert\, l_2\bigr)
        \;=\;
        \sum_{f\in\mathcal{F}}
        P_{l_1}(f)\,
        \log\!\frac{P_{l_1}(f)}{P_{l_2}(f)}
    $$
- **Magnitude-Squared Coherence**  
    $$
        C_{xy}(f)=\frac{|G_{xy}(f)|²}{G_{xx}(f) G_{yy}(f)}
    $$

---
<!-- header: 'Experimental Pipeline' -->
1. **Datasets**: Bible-Corpus (100 langs), XNLI (15 langs)  
2. Encode with mBERT & XLM-R  
3. Extract token likelihoods/embeddings  
4. Compute PSD via Welch  
5. Compute Overlap, KL, Coherence for every pair  
6. Correlate with 4 baselines

---
<!-- header: 'Table of contents' -->
Motivation
Research questions & contributions  
Background 
Methodology   
**Experiments**  
Results  
Discussion & limitations  
Future work

---
<!-- header: 'Experiments' -->
- **E1** Time- vs. frequency-domain signals  
- **E2** Model robustness (mBERT/XLM-R)  
- **E3** Dataset robustness (Bible/XNLI)  
- **E4** Layer-wise probe (12 layers)  
- **E5** Sliding frequency windows (.04 & .10 bandwidth)  
- **E6** WALS feature probing

---
<!-- header: 'Table of contents' -->
Motivation
Research questions & contributions  
Background 
Methodology   
Experiments  
**Results**  
Discussion & limitations  
Future work

---
<!-- header: 'Key Results' -->
| Baseline | Best r | Metric / Signal |
|----------|-------:|-----------------|
| Phonemic | **0.62** | Coherence / embed |
| Lexical  | **0.67** | Coherence / embed |
| Intellig. | **–0.54** | Coherence / embed |
| FSI weeks | **0.57** | Spectral KL / embed |

* Two hot bands: **4–8 %** & **58–80 %** of Nyquist

---
<!-- header: 'Table of contents' -->
Motivation
Research questions & contributions  
Background 
Methodology   
Experiments  
Results  
**Discussion & limitations**  
Future work

---
<!-- header: 'Strengths' -->
* Annotation-free & scalable (100 + langs)  
* Integrates multiple linguistic layers  
* Stable across models, corpora, layers

---
<!-- header: 'Limitations' -->
* Frequency spikes are indirect to interpret  
* May be sensitive to tokenization  
* KL asymmetry too mild for intelligibility gaps  
* Few large parallel corpora

---
<!-- header: 'Table of contents' -->
Motivation
Research questions & contributions  
Background 
Methodology   
Experiments  
Results  
Discussion & limitations  
**Future work**

---
<!-- header: 'Future Work' -->
* Add syntactic baselines  
* Compare time-domain cross-correlation  
* Test more genres & dialect continua  
* Try **cepstral** analysis for harmonic structures  
* Design strongly asymmetric spectral metrics

---
<!-- header: 'Take-Home Message' -->
**Spectral fingerprints of multilingual-transformer signals give an annotation-free and faithful measure of linguistic distance, bridging neural representations with classical typology.**


