This set is in good shape — the fix has actually propagated correctly everywhere it needed to. A few things worth confirming explicitly, plus one place where I can close out something you've explicitly left open.

**The core fix checks out, and you found stronger evidence than I had**

A.6's injection-convention note is better grounding than my LAL-docs search from before — you traced `coa_phase` through your *own* `gen.py` straight into `pycbc.waveform.get_td_waveform(coa_phase=...)`, confirming it's literally PyCBC/LAL's `phiRef`. That's a direct confirmation for your specific dataset, not just "LAL generally documents it this way." Combined with IMRPhenomD being (2,2)-only, the "dominant quadrupole" assumption in every formula here isn't even an approximation for this dataset — it's exact for the generating model. A.2, A.5, and A.8 are all internally consistent with `2φc±2ψ`, and nothing in `redo_procedure.md` reintroduces the old formula anywhere I can find (§1.1, §1.2, §2.3, §2.6 all correctly redo the steps that depended on it, and §1.3–1.5 correctly leave alone the parts that didn't).

**A.5's required self-check — I ran it through by hand, it passes**

Worth confirming rather than taking on faith, since it's the gate for everything downstream. At $\iota=0$: $k_1=1,k_2=1$, and

$$\text{detector\_signal} = F_+\cos\Theta + F_\times\sin\Theta = a\cos(\Theta+2\psi)+b\sin(\Theta+2\psi),\quad \Theta=2\Phi+2\varphi_c$$

so the signal depends on $\varphi_c,\psi$ only through $\Delta\equiv2\varphi_c+2\psi$ — exactly the invariance your self-check demands. Good gate to have in place before `derive_w_iota` runs.

(Small aside, not a bug: your toy $F_+,F_\times$ pair transforms as $e^{+2i\psi}$ rather than $e^{-2i\psi}$ — opposite handedness from the convention I used earlier — but the ratio $k_1/k_2=(1+\cos^2\iota)/(2\cos\iota)$ matches the true physical amplitude ratio exactly, and the self-check above passes regardless of that handedness choice, so it washes out. Nothing to fix.)

**A.8 / §2.6 — you flagged this as needing re-derivation before writing code. Here it is.**

Let $U=2\varphi_c$ (range $[0,4\pi)$, since $\varphi_c$'s own physical period is $2\pi$) and $V=2\psi$ (range $[0,2\pi)$, since $\psi$'s own physical period is $\pi$ — note $V$ alone is already a *bijection* onto $\psi$'s full physical range, no ambiguity yet). From `combo_A`$=U+V$ and `combo_B`$=U-V$ (both mod $2\pi$):

$$2V \equiv \text{combo\_A}-\text{combo\_B} \pmod{2\pi} \;\Rightarrow\; V \equiv \tfrac12(\text{combo\_A}-\text{combo\_B}) \pmod \pi$$

That's a 2-fold ambiguity in $V$'s own $2\pi$ domain, which maps 1:1 to a **2-fold** ambiguity in $\psi$ (candidates $\psi_0,\ \psi_0+\pi/2$) — not 4-fold. Similarly for $U$:

$$2U \equiv \text{combo\_A}+\text{combo\_B}\pmod{2\pi} \;\Rightarrow\; U\equiv\tfrac12(\text{combo\_A}+\text{combo\_B})\pmod\pi$$

giving a **4-fold** ambiguity in $U$'s $4\pi$ domain, mapping 1:1 to 4 candidates in $\varphi_c$ (spaced $\pi/2$ apart) — this part matches what A.8 already says.

The part actually worth flagging: these two ambiguities are **not independently combinable**. Parametrize candidates as $\varphi_c=\varphi_{c,0}+k\tfrac\pi2$ ($k=0,1,2,3$), $\psi=\psi_0+j\tfrac\pi2$ ($j=0,1$). Reconstructing `combo_A`,`combo_B` from a given $(k,j)$ reproduces the *original* pair only when $k$ and $j$ have matching parity — both even or both odd (this falls out because $k+j$ and $k-j$ always share parity). That keeps 4 of the naive $4\times2=8$ combinations, not 8, and not a clean "4-fold × 2-fold." So the mandatory filter in A.8 isn't just a safety check — it's load-bearing: it's what cuts the joint ambiguity from 8 down to the true 4. Worth writing that parity rule directly into the reconstruction code as a shortcut rather than brute-force filtering all 8 every time, though the brute-force filter as a validation cross-check is still the right call to keep alongside it.