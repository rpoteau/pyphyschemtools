# kinetics

## Kinetic Order of Reaction Determination, KORD

* [1. Principle](#principle)
* [2. Theoretical Model](#theoretical-model)

---

<a id="principle"></a>
### 1. Principle

#### 1.1 Experimental Measurements ($G_\mathrm{EXP}$)

<div class="intro">
    
In chemical kinetics, we track the evolution of molar concentrations over time: $C_{i}(t)$. Experimentally, we measure a physical quantity proportional to these concentrations: $G_\mathrm{EXP}(t)$.

* **Spectrophotometry**: $G_\mathrm{EXP} = A = \sum_{i} \epsilon_{i} \cdot l \cdot C_{i}$, where $A$ is the absorbance
* **Polarimetry**: $G_\mathrm{EXP} = \alpha = \sum_{i} [\alpha]_{i} \cdot l \cdot C_{i}^{w}$
* **Conductivity**: $G_\mathrm{EXP} = \sigma = \sum_{i} \lambda_{i} \cdot C_{i}$

**General Form:** $G_\mathrm{EXP}(t) = \sum_{i} \eta_{i} \cdot C_{i}(t)$

</div>

<a id="theoretical-model-principle"></a>
#### 1.2 Theoretical Model ($G_\mathrm{THEO}$)

<div class="intro">

The model $G_\mathrm{THEO}$ is defined by two types of values: **fixed parameters** (input by the user) and **adjustable variables** (optimized by the algorithm).

**<u>Fixed Parameters</u>:**
* **Reaction Order**: $n \in \{0, 1, 2\}$ (The user selects the order to test).
* **Stoichiometry**: $\alpha$ and $\beta$ are known constants, provided by the user.
  <div class="rqE">
      
  $\alpha$ and $\beta$ must be the smallest possible positive integers
  </div>
* **Initial Concentration**: $A_{0}$ (Note: For Order 1, $G_{THEO}$ is independent of $A_{0}$). The concentration must also be provided by the user

  <div class="rqE">
        
  The final concentration of B, $B_{\infty}$ is related to $A_0$ by the relation:

  $$\frac{A_0}{\alpha} = \frac{B_{\infty}}{\beta}$$
  </div>

**<u>Adjustable Variables</u>:**
The model fits the experimental data by adjusting the following:
* **Rate Constant**: $k$
* **Final Value**: $G_{\infty}$
* **Initial Value**: $G_{0}$ (While $G_{0}$ is measured, the algorithm also adjusts it to ensure the best fit starting point).

The optimization is performed for a specific reaction order at a time to determine which model best describes the experimental data.

<div class="rqE">

By default, KORD chooses the first and last $G_\mathrm{EXP}$ values as $G_{0}$ and $G_{\infty}$. And a default $k$ value is also setup by KORD. If you need to change that because of a convergence issue, ensure your starting values for $k$, $G_{0}$ and $G_{\infty}$ are realistic to help the algorithm converge. 

</div>

</div>

#### 1.3 Optimization (RMSD)

<div class="intro">

The algorithm minimizes the **Root-Mean-Square Deviation** to fit the theoretical curve to the experimental data:

$$RMSD = \sqrt{\frac{1}{n} \sum_{k=1}^{n} \{G_\mathrm{EXP}(t_{k}) - G_\mathrm{THEO}(t_{k})\}^{2}}$$

</div>

#### 1.4 Input

<div class="intro">

Data input is performed through a structured Excel file. Users simply provide the kinetic parameters ($\alpha$, $\beta$), the initial concentration $[A]_0$, and the experimental data series (time $t$ and property $G_{\mathrm{exp}}$). 

</div>


---

<a id="theoretical-model"></a>
### 2. Theoretical Model

<div class="intro">

The reaction model used in KORD is designed to be as simple as possible based on the following criteria:

- Single-component reaction: A unique reactant $A$ transforms into a unique product $B$ ($\alpha A \rightarrow \beta B$)
- Total reaction: The reaction goes to completion (the extent of reaction is 100%)
- Closed system: No exchange of matter occurs between the system and its environment; only energy exchanges are possible
- Homogeneous system: The concentration of any compound $C_i$ is uniform throughout the entire system
- Isochoric system: The volume of the system remains constant throughout the reaction.

</div>

#### 2.1 Reactant Expression $A(t)$

<div class="intro">

The rate law is defined as:

$$v = -\frac{1}{\alpha} \frac{dA}{dt} = k A^{n}$$

* **Order 0**: 

$$A(t)=A_{0}-\alpha kt$$

* **Order 1**: 

$$A(t) = A_{0} \exp(-\alpha kt)$$

* **Order 2**: 

$$A(t) = \frac{1}{\frac{1}{A_{0}} + \alpha kt}$$

</div>

#### 2.2 Product Expression $B(t)$

<div class="intro">

Derived from mass balance ($M_{A} A(t) + M_{B} B(t) = M_{A} A_{0} = M_{B} B_{\infty}$):
* **Order 0**:

$$B(t)=\beta kt$$

* **Order 1**:

$$B(t) = B_{\infty} \{1 - \exp(-\alpha kt)\}$$

* **Order 2**:

$$B(t)=B_{\infty}\left(\frac{1}{1+\frac{\alpha}{A_{0}\alpha^{2}kt}}\right)$$

</div>

#### 2.3 Global Expression $G_\mathrm{THEO}(t)$

<div class="intro">

The theoretical quantity is a linear combination of $A(t)$ and $B(t)$:

$$G_\mathrm{THEO}(t) = \frac{G_{0}}{A_{0}} \cdot A(t) + \frac{G_{\infty}}{B_{\infty}} \cdot B(t)$$

**Standard Formula for Order 0**:

$$G_{\mathrm{THEO}}(t)=G_{0}+\frac{\alpha kt}{A_{0}}(G_{\infty}-G_{0})$$

<div class="rqE">
    
Warning: This mathematical model for Order 0 is a linear equation. Unlike Order 1 or 2, this linear model does not naturally plateau. Depending on the values of $k$ and $t$, the model may predict non-physical values (e.g., negative absorbance or negative concentration) if the time $t$ exceeds the theoretical completion time $t_{end} = \frac{A_{0}}{\alpha k}$. These values are mathematical artifacts and should be ignored.
    
</div>
<br>

**Standard Formula for Order 1**:

$$G_\mathrm{THEO}(t) = G_{\infty} + \exp(-\alpha kt)(G_{0} - G_{\infty})$$

**Standard Formula for Order 2**:

$$G_{\mathrm{THEO}}(t)=G_{\infty}-\frac{1}{1+A_{0}\alpha kt}(G_{\infty}-G_{0})$$

</div>

