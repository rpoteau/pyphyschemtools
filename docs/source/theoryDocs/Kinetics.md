# Chemical Kinetics

**`kinetics.py` module**

So far, it only contains the `KORD` class

## Kinetic Order of Reaction Determination, KORD

* [1. Principle](#principle)
* [2. Theoretical Model](#theoretical-model)

---

<a id="principle"></a>
### 1. Principle

#### 1.1 Experimental Measurements ($G_\mathrm{EXP}$)

In chemical kinetics, we track the evolution of molar concentrations over time: $C_{i}(t)$. Experimentally, we measure a physical quantity proportional to these concentrations: $G_\mathrm{EXP}(t)$.

* **Spectrophotometry**: $G_\mathrm{EXP} = A = \sum_{i} \epsilon_{i} \cdot l \cdot C_{i}$, where $A$ is the absorbance
* **Polarimetry**: $G_\mathrm{EXP} = \alpha = \sum_{i} [\alpha]_{i} \cdot l \cdot C_{i}^{w}$
* **Conductivity**: $G_\mathrm{EXP} = \sigma = \sum_{i} \lambda_{i} \cdot C_{i}$

**General Form:** $G_\mathrm{EXP}(t) = \sum_{i} \eta_{i} \cdot C_{i}(t)$

<a id="theoretical-model-principle"></a>
#### 1.2 Theoretical Model ($G_\mathrm{THEO}$)

The model has three different expressions (see <a href="#theoretical-model">Theoretical model section</a>), depending on the order of the reaction:

**Standard Formula for Order 0**:

$$\hat{G}^{[0]}(t)=\begin{cases}
G_{0}+\beta\dfrac{kt}{b_{\infty}}\left(G_{\infty}-G_{0}\right) & t\leq\dfrac{a_{0}}{\alpha k}=\dfrac{b_{\infty}}{\beta k}\\
G_{\infty} & t>\dfrac{a_{0}}{\alpha k}=\dfrac{b_{\infty}}{\beta k}
\end{cases}$$

**Standard Formula for Order 1**:

$$\hat{G}^{[1]}(t)=G_{\infty}+\exp\left(-\alpha kt\right)\left(G_{0}-G_{\infty}\right)$$

**Standard Formula for Order 2**:

$$\hat{G}^{[2]}(t)=
G_{0}+\dfrac{\left(G_{\infty}-G_{0}\right)}{1+\dfrac{\beta}{b_{\infty}\alpha^{2}kt}}=G_{0}+\frac{\left(G_{\infty}-G_{0}\right)b_{\infty}\alpha^{2}kt}{b_{\infty}\alpha^{2}kt+\beta}$$


$G_\mathrm{THEO}$ is defined by two types of values: **fixed parameters** (input by the user) and **adjustable variables** (optimized by the algorithm).

**<u>Fixed Parameters</u>:**
* **Reaction Order**: $n \in \{0, 1, 2\}$ (The user selects the order to test).
* **Stoichiometry**: $\alpha$ and $\beta$ are known constants, provided by the user.
      
  $\alpha$ and $\beta$ must be the smallest possible positive integers

* **Initial Concentration**: $a_{0}$ (Note: For Order 1, $G_{THEO}$ is independent of $a_{0}$). The concentration must also be provided by the user

  The final concentration of B, $b_{\infty}$ is related to $a_0$ by the relation:

  $$\frac{a_0}{\alpha} = \frac{b_{\infty}}{\beta}$$

**<u>Adjustable Variables</u>:**
The model fits the experimental data by adjusting the following:
* **Rate Constant**: $k$
* **Final Value**: $G_{\infty}$
* **Initial Value**: $G_{0}$ (While $G_{0}$ is measured, the algorithm also adjusts it to ensure the best fit starting point).

The optimization is performed for a specific reaction order at a time to determine which model best describes the experimental data.

<div class="rqE">

By default, KORD chooses the first and last $G_\mathrm{EXP}$ values as $G_{0}$ and $G_{\infty}$. And a default $k$ value is also setup by KORD. If you need to change that because of a convergence issue, ensure your starting values for $k$, $G_{0}$ and $G_{\infty}$ are realistic to help the algorithm converge. 

</div>

#### 1.3 Optimization (RMSD)

The algorithm minimizes the **Root-Mean-Square Deviation** to fit the theoretical curve to the experimental data:

$$RMSD = \sqrt{\frac{1}{n} \sum_{k=1}^{n} \{G_\mathrm{EXP}(t_{k}) - G_\mathrm{THEO}(t_{k})\}^{2}}$$


#### 1.4 Input

Data input is performed through a structured Excel file. Users simply provide the kinetic parameters ($\alpha$, $\beta$), the initial concentration $[A]_0$, and the experimental data series (time $t$ and property $G_{\mathrm{exp}}$). 

---

<a id="theoretical-model"></a>
### 2. Theoretical Model

The reaction model used in KORD is designed to be as simple as possible based on the following criteria:

- Single-component reaction: A **unique reactant** $A$ transforms into a **unique product** $B$ ($\alpha A \rightarrow \beta B$)
- **Total reaction**: The reaction goes to completion (the extent of reaction is 100%)
- **Closed system**: No exchange of matter occurs between the system and its environment; only energy exchanges are possible
- **Homogeneous system**: The concentration of any compound $C_i$ is uniform throughout the entire system
- **Isochoric system**: The volume of the system remains constant throughout the reaction.

#### 2.1 Reactant Expression $a(t)$

The rate law is defined as:

$$v = -\frac{1}{\alpha} \frac{d[A]}{dt} = k [A]^{n}$$

* **Order 0**: 

$$a(t)=\begin{cases}
a_{0}-\alpha kt & t\leq\frac{a_{0}}{\alpha k}\\
0 & t>\frac{a_{0}}{\alpha k}
\end{cases}$$

* **Order 1**: 

$$a(t) = a_{0} \exp(-\alpha kt)$$

* **Order 2**: 

$$a(t) = \frac{1}{\frac{1}{a_{0}} + \alpha kt}$$

#### 2.2 Product Expression $B(t)$


Derived from mass balance ($M_{A} a(t) + M_{B} b(t) = M_{A} a_{0} = M_{B} b_{\infty}$):
* **Order 0**:

$$b(t)=\begin{cases}
\beta kt & t\leq\frac{b_{\infty}}{\beta k}\\
0 & t>\frac{b_{\infty}}{\beta k}
\end{cases}$$

* **Order 1**:

$$b(t) = b_{\infty} \{1 - \exp(-\alpha kt)\}$$

* **Order 2**:

$$b(t)=\frac{b_{\infty}}{1+\frac{\beta}{b_{\infty}\alpha^{2}kt}}$$


#### 2.3 Global Expression $G_\mathrm{THEO}(t)$


The theoretical quantity is a linear combination of $a(t)$ and $b(t)$:

$$G_\mathrm{THEO}(t) = \frac{G_{0}}{a_{0}} \cdot a(t) + \frac{G_{\infty}}{b_{\infty}} \cdot b(t)$$

**Standard Formula for Order 0**:

$$\hat{G}^{[0]}(t)=\begin{cases}
G_{0}+\beta\dfrac{kt}{b_{\infty}}\left(G_{\infty}-G_{0}\right) & t\leq\dfrac{a_{0}}{\alpha k}=\dfrac{b_{\infty}}{\beta k}\\
G_{\infty} & t>\dfrac{a_{0}}{\alpha k}=\dfrac{b_{\infty}}{\beta k}
\end{cases}$$

<div class="rqE">
    
Warning: This mathematical model for Order 0 is a linear equation. Unlike Order 1 or 2, this linear model does not naturally plateau. Depending on the values of $k$ and $t$, the model may predict non-physical values (e.g., negative absorbance or negative concentration) if the time $t$ exceeds the theoretical completion time $t_{\mathrm{end}} = \frac{a_{0}}{\alpha k}=\frac{b_{\infty}}{\beta k}$. These values are mathematical artifacts and should be ignored.
    
</div>
<br>

**Standard Formula for Order 1**:

$$\hat{G}^{[1]}(t)=G_{\infty}+\exp\left(-\alpha kt\right)\left(G_{0}-G_{\infty}\right)$$

**Standard Formula for Order 2**:

$$\hat{G}^{[0]}(t)=\begin{cases}
G_{0}+\beta\dfrac{kt}{b_{\infty}}\left(G_{\infty}-G_{0}\right) & t\leq\dfrac{a_{0}}{\alpha k}=\dfrac{b_{\infty}}{\beta k}\\
G_{\infty} & t>\dfrac{a_{0}}{\alpha k}=\dfrac{b_{\infty}}{\beta k}
\end{cases}$$


