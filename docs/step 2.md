# Executive summary / 执行摘要

The second stage (Step 2) of the pipeline synthesizes information from all collected per-paper extraction records into a single comprehensive report.  Its goal is to **integrate evidence, patterns, and assumptions across papers** in a way that remains neutral to any one system architecture.  In practice, Step 2 “involves collating and summarising the results of the included primary studies”【3†L2802-L2810】, aiming to increase the generality of findings【23†L83-L91】.  A well-designed Step 2 report clearly separates facts from interpretation, annotates confidence and uncertainty, and preserves contradictions or open questions.  It highlights recurring mechanisms and trade-offs without committing to a particular design choice.  The output is a structured, layered artifact (e.g. executive-summary + technical detail + evidence appendices) that supports multiple downstream designs.  We recommend a **multi-pass workflow** (data normalization, cross-paper mapping, contradiction analysis, etc.) and a **structured report template** (e.g. clearly sectioned Markdown with explicit links to evidence).  These practices ensure neutrality, traceability, and reusability across different design branches.  For example, interactive evidence-mapping research shows that conventional narrative reviews can “encourage overgeneralization” and hide critical gaps【20†L76-L85】; our Step 2 report will explicitly surface such gaps and label uncertainty.  By analogy with Systematic Evidence Maps (SEMs), Step 2 should use structured tables and diagrams to categorize evidence and identify trends【13†L78-L87】【3†L2824-L2831】, enabling engineers to explore the synthesized knowledge with confidence.

步骤2的核心在于从多个论文提取记录中合并信息，生成一个综合性的、对具体系统架构保持中立的研究报告。它的目标是**整理并综合各项证据、模式和假设**，而不是直接提出解决方案。实际上，步骤2“涉及将所纳入研究的结果进行汇总和总结”【3†L2802-L2810】，其目标是提高发现结果的普遍性和适用性【23†L83-L91】。一个高质量的步骤2报告应当清晰区分事实与解读，对结论标注置信度和不确定性，并保留研究间的矛盾或未解决的问题。报告需要突出重复出现的机制和折衷方案，但不偏向任何特定设计选择。最终输出是一份结构化的文件（例如，执行摘要+技术细节+证据附录分层呈现），能够支持不同的后续设计方案。我们建议采用**多轮工作流程**（数据规范化、跨论文映射、矛盾分析等）和**结构化报告模板**（如明确分节的Markdown格式，并与原始证据条目链接）。这些做法能确保中立性、可追溯性和跨设计分支的可复用性。例如，交互式证据地图研究指出传统叙述性综述“易促使发现过度泛化，并掩盖文献中的重要空白”【20†L76-L85】；我们的步骤2报告将显式揭示这些空白并标注不确定性。借鉴系统性证据图谱（SEM），步骤2应使用表格和图示对证据进行分类和趋势分析【13†L78-L87】【3†L2824-L2831】，从而帮助工程师有效探索综合信息。

# Why Step 2 must be separated from Step 3 / 为什么需要将步骤2与步骤3分开

**Step 2’s role** is to gather and interpret knowledge, not to choose solutions.  By separating Step 2 from Step 3 (system design), we **prevent premature commitment** to any specific architecture.  In research terms, Step 2 is akin to a systematic literature synthesis that “deals with the challenge of information overload” and distills evidence for decision-making【23†L83-L92】.  If we mix Step 2 with design (Step 3) too early, the synthesis can become biased: authors may cherry-pick or spin results to justify a favored design.  This not only reduces the reuse of the synthesis (future designs will be skewed) but also hides uncertainties and alternative approaches.  For example, Wyborn et al. note that there is “no single ‘correct way’ to design research synthesis for impact,” implying the approach must be tailored to context【23†L97-L104】.  In practice, Step 2 should follow rigorous, pre-planned methods (as in systematic reviews) to minimize bias【42†L178-L187】, and leave the actual architectural decisions for Step 3.

步骤2的目的在于归纳和解释知识，而不是选择最终的系统方案。将步骤2与步骤3（系统设计）分离，可以**避免过早地锁定某个特定架构**。从研究角度看，步骤2类似于系统综述，能够应对“信息过载”的挑战，并为决策提炼证据【23†L83-L92】。如果过早混入设计步骤，综合过程可能受到偏见影响：研究人员可能选择性地引用或解读信息来支持某种设计方案。这不仅削弱了综合报告的可复用性（未来设计将受到偏见影响），而且也隐藏了不确定性和其它可行方案。正如Wyborn等人指出，**“设计研究综合以影响为目的并没有唯一正确的方式”**【23†L97-L104】，说明方法需要根据目标情境定制。在实践中，步骤2应遵循系统综述中那种事先规划好的、系统的方法来尽量减少偏差【42†L178-L187】，而将具体的架构抉择留给步骤3去处理。

# Failure modes of weak or architecture-biased Step 2 reports / 弱化或带有架构偏见的步骤2报告的失败模式

If Step 2 is not carefully controlled, several failure modes emerge:

- **Bias from premature design:** If the report effectively chooses an architecture too early, it will *only* present evidence supporting that design. Contradictions and alternatives get downplayed or ignored. This locks the pipeline into one branch and wastes the synthesis on a predetermined solution. In literature-review terms, this is akin to a narrative review that “缺乏规范的检索和筛选方法，结果存在偏见，不彻底”【42†L228-L236】.
- **Loss of nuance and contradiction:** A weak synthesis may focus only on ‘consistent’ findings. It risks the “overgeneralization” problem identified by Mallavarapu et al.: emphasizing broad conclusions at the expense of nuanced details and gaps【20†L76-L85】. Important disagreements between papers may be brushed under the rug.
- **Lack of traceability:** If claims in the report are not explicitly linked to source papers, credibility suffers. Designers cannot verify which fact came from which study, making the report a “black box”. This also makes it impossible to update or expand the report reliably.
- **Narrative fluff and missing structure:** Without strict structure, the report can devolve into vague prose. Traditional narrative reviews are known to be “含有偏见、不 exhaust...的结论”【42†L228-L236】. A report lacking clear sections (e.g. no taxonomy, no evidence tables) will be hard to use programmatically and easy to misinterpret.
- **Mixing analysis with design:** Including performance estimates, hardware choices, or interface designs in Step 2 leads to bias. Such content should belong in Step 3. Mixing it early means Step 2 is no longer architecture-neutral.

总之，弱化的步骤2报告会出现多种问题：**过早的设计偏见**会导致报告仅呈现支持特定方案的证据，而忽略矛盾和其他可能性，这相当于传统叙述性综述“缺乏规范的检索和筛选方法，结果存在偏见，不彻底”【42†L228-L236】。**细节和矛盾的丢失**是另一个风险：正如Mallavarapu等指出，注重综合会“鼓励对发现结果的过度泛化，并掩盖文献中的重要空白”【20†L76-L85】。**可追溯性缺失**意味着报告中的结论无法与原始研究对应，降低了可信度，也无法方便地更新或扩展。**缺乏结构或过于叙述**的报告难以程序化使用，容易被误解，正如上文所引叙述性综述的批评【42†L228-L236】。最后，如果将特定的技术推断混入报告（例如性能指标或硬件选型），步骤2就失去了架构中立性，因为这些内容本应留到步骤3。因此，我们必须设计严格的步骤2流程，避免上述失败模式。

# Design principles for Step 2 / 步骤2的设计原则

Step 2 should be guided by the following principles:

- **Evidence-focused and systematic:** Follow a pre-defined protocol to gather and synthesize data.  Like a systematic review, it should use explicit methods to minimize bias【42†L178-L187】.  For example, Step 2 should define a clear scope and screening criteria, then uniformly apply them to all Step 1 records【13†L84-L91】.
- **Design-neutral and multi-perspective:** Do not encode any specific architectural preference.  Present all options and trade-offs uniformly. Maintain parallel tracks (e.g. cloud vs on-prem, centralized vs distributed) without favoring one.
- **Transparent traceability:** Every claim or summary point should be linked to its source paper(s). Use consistent reference codes so readers can trace back details. This is analogous to PRISMA/checklist best practices in reviews【42†L235-L244】, ensuring decisions are justified by cited evidence.
- **Structured and modular:** Organize content into clear sections and tables. For example, collate key attributes of each approach into comparison tables【3†L2824-L2831】. Use taxonomies to categorize approaches (e.g. by algorithm type or use-case). This aligns with evidence-mapping advice to categorize evidence and highlight trends【13†L78-L87】【3†L2824-L2831】.
- **Explicit uncertainty and contradictions:** Label items as “observations”, “hypotheses”, “open questions”, or “conflicts”. Where evidence is weak or contradictory, explicitly note the lack of consensus. We may adopt approaches from GRADE/CERQual to annotate confidence【29†L11-L14】. For example, mark findings as “high confidence” if multiple studies agree, or “low confidence” if evidence is sparse.
- **Actionable insight without commitment:** Step 2 should highlight design implications (e.g. scalability trade-offs, operational constraints) but stop short of choosing a solution. It’s like generating an unbiased options analysis. Any decision suggestions should be couched as conditional (“If requirement X, consider mechanisms A or B”) rather than declarative.

这些原则可总结为**系统化、透明化、中立化**。具体而言，步骤2应遵循事先定义的协议，使用系统的方法来汇总数据，以尽量减少偏见【42†L178-L187】。比如，应明确设计范围和筛选条件，并统一应用于所有步骤1记录【13†L84-L91】。同时，报告保持设计中立（不偏袒任何架构）并从多角度展示信息，例如分别讨论“云端 vs 本地”、“中心化 vs 分布式”等不同场景，而不预先选边站队。报告中每条结论或观点都应引用来源论文（类似PRISMA指南所强调的做法【42†L235-L244】），确保可追溯。例如，可以使用一致的编号标注每条发现来自哪篇论文。内容上要**结构化和模块化**：可使用表格来对比各方案的关键特性【3†L2824-L2831】，使用分类法对方法类型进行归类【13†L78-L87】【3†L2824-L2831】。对于不确定或相互矛盾的信息，要明确标注：例如引用GRADE/CERQual方法为结论打上“高置信度”或“低置信度”标签【29†L11-L14】。最后，步骤2只提供可操作的见解，而不做最终决策。比如，根据发现指出各方案的可行性或风险，但不要直接推荐某种具体实现。这样，步骤2报告会为后续的架构设计留出充分的选择空间。

# Required goals of a Step 2 research synthesis report / 步骤2研究综合报告的必备目标

A successful Step 2 report should achieve the following goals:

- **Comprehensive evidence synthesis:** Integrate all relevant findings from the Step 1 records. The report should “distill relevant evidence for decision-making”【23†L83-L92】 by summarizing what each approach has achieved (performance, assumptions, limitations). It must preserve the **heterogeneity** of results: when studies differ, the report should note the divergence rather than oversimplifying.
- **Preserve assumptions and context:** Explicitly capture the assumptions and preconditions underlying each method (e.g. dataset size, failure modes, operational environment). These might not be obvious from numbers but are critical for design. An “Assumption Map” section can list these by theme.
- **Highlight contradictions and gaps:** Flag any inconsistent or conflicting results across papers. Where the literature disagrees, Step 2 should **not** force a resolution; instead, it should document the contradiction and note it as an open question. This gives designers insight into risk and uncertainty.
- **Identify recurring patterns:** Detect repeated mechanisms, patterns, or design motifs (e.g. common heuristics, algorithms, or architecture styles). Similarly, list **reusable components** or primitives gleaned from multiple sources (forming an inventory of engineering patterns). This helps jumpstart later design reuse.
- **Map evidence strength and uncertainty:** Provide an evidence-strength map (or table) that shows how robust each finding is. For instance, tag items supported by multiple independent studies as high-confidence, while one-off claims are low-confidence. This mirrors evidence grading practices【29†L11-L14】.
- **Outline operational implications:** Translate key findings into implications for real-world deployment (e.g. scalability limits, expected maintenance challenges). If possible, include a brief **production-readiness assessment** (e.g. proven in prototypes vs only theoretical).
- **Leave no design decisions:** Refrain from proposing a specific system architecture or tuning parameters. Any discussion of “design implications” should be framed as considerations, not choices.

总的来说，步骤2报告要**整合所有步骤1的相关发现**，“为决策者提炼出相关证据”【23†L83-L92】。这意味着要系统总结每种方法的成果（性能、假设和局限性），并保留研究结果的**多样性**：当不同研究结论不同时，报告应记录这种差异，而不是模糊化。例如，应该包含一个“假设映射”部分，列出每种方法依赖的前提条件和假设。报告还要**突出矛盾点和空白**：如果文献对同一问题存在分歧，步骤2应记录这种矛盾并将其视为未解问题，不应强行达成一致，这样设计者可以意识到潜在风险。相反，应主动识别**重复出现的模式和可重用原语**：例如，将跨论文常见的算法或架构片段编入一个组件清单。这为后续设计工作提供了宝贵的参考。同样，还要给出**证据强度图**（或表格），标注每个结论的置信度，例如多篇研究一致支持的结论为高可信度【29†L11-L14】，单篇报告为低可信度。另外，需总结每个重要发现的**实际应用影响**，如可扩展性限制或维护难点，并简要评估生产可行性（如已有原型验证 vs 仅有理论）。但需要注意的是，这些都只是为设计提供参考。报告本身**不应提出确定的系统设计决策**，任何“设计指导”都应以讨论性质呈现，而非定论。

# Recommended information architecture / 建议的信息架构

A high-quality Step 2 report should have clear, logical sections. Essential sections include:

- **Executive Summary / 关键发现:** A brief overview of the most important insights and recommendations (written neutrally), to orient decision-makers. It answers “What patterns and conclusions emerged across the papers?” without technical detail.
- **Scope and Corpus Definition / 研究范围:** Precisely describe the problem domain, research questions, and criteria that guided paper selection. This anchors the report and prevents drifting into irrelevant areas.
- **Methodology / 综合方法:** Explain how the synthesis was done (e.g. number of records, normalization steps, etc.). This enhances trust by making the process transparent.
- **Taxonomy of Approaches / 方法分类:** Classify the different solutions/approaches found in Step 1. For example, group papers by algorithmic paradigm, deployment model, or system component. This helps designers quickly navigate the solution space.
- **Recurring Mechanisms and Patterns / 机制与模式:** Describe repeatedly observed principles or components (e.g. “peer-to-peer gossip” or “transactional consistency layer”). This reveals common building blocks.
- **Assumption Map / 假设映射:** List the key assumptions from all studies (e.g. “assumes independent failures” or “requires full reliability”). Each assumption should cite supporting papers. This section prevents oversight of hidden conditions.
- **Contradiction Map / 冲突图谱:** Explicitly present any conflicts in the literature (e.g. “Method A reports scalability to 1000 nodes, but Method B on the same problem reports only 100 nodes”). Use a table or bullet list format. Highlight how many papers support each side. This guards against overlooking disagreements.
- **Evidence Strength Map / 证据强度图:** Summarize the confidence in each synthesized finding (e.g. heatmap or graded list). For instance, indicate which performance claims are validated by multiple experiments vs single case studies.
- **Operational Implications / 运营影响:** Translate findings into engineering terms: performance bottlenecks, reliability considerations, cost factors, etc. For example, note if a method relies on assumptions that are hard to meet in real deployments.
- **Production-readiness Assessment / 生产准备度:** For each approach or pattern, rate its maturity (e.g. prototype, open-source, industrial use). This informs Step 3 designers about potential engineering effort.
- **Reusable Mechanism Inventory / 可重用组件清单:** Catalog specific modules or algorithms identified that can be reused (possibly cross-referenced with implementation references).
- **Design Implications / 设计启示:** Discuss how the synthesized knowledge might influence architecture (e.g. “To achieve required throughput, designs should consider bulk-loading optimization like Method C”). These notes stop short of prescribing a full design.
- **Unresolved Questions / 悬而未决的问题:** Clearly enumerate open issues or gaps (e.g. “No study addresses latency under cross-data-center deployment”). This highlights future research needs and things to verify in design.
- **Risk Register / 风险登记:** List potential risks inferred from the synthesis (technical risks, assumption failures, etc.), possibly mapping to findings.
- **Traceability Appendix / 可追溯性附录:** A machine-readable appendix that ties every assertion in the report back to specific Step 1 records or citations (e.g. mapping table of statement→source).

Sections *critical for reuse* include the taxonomy (to scope solutions), patterns/mechanisms, assumptions, and contradictions – these preserve the raw knowledge needed by multiple designs. Sections *critical for neutrality* include the contradiction/uncertainty map and evidence strength map – they explicitly surface disagreements and confidence levels, preventing unconscious bias. Executives often skip fine detail, so missing evidence tables is common in ordinary reviews, but here a structured evidence table is crucial. Likewise, assumption maps are often omitted in typical literature summaries, yet they are vital to prevent hidden biases in design. Each recommended section serves a question: *What are the solutions (taxonomy)?* *What conditions do they assume?* *Where do they conflict?* *How strong is each finding?* *What engineering effects arise?* etc. Omitting, say, contradiction analysis would answer none of those and harm downstream decision-making.

# Evidence, uncertainty, and traceability model / 证据、不确定性和可追溯性模型

To ensure rigor, Step 2 must clearly distinguish evidence from interpretation and track everything to sources:

- **Traceable citations:** Every factual claim, metric, or quote in the report should be annotated with a reference to the specific Step 1 record (and original paper) from which it was derived. This ensures transparency and allows readers or tools to drill down. For example, if we state “Method X achieved 95% accuracy on Y,” we tag it like 【PaperID†Lxx-Lyy】 linking to the extraction record of that paper. The appendix will cross-link these IDs. This level of traceability is akin to audit trails in systematic reviews.
- **Evidence-backed vs. inference:** Use notation or phrasing to separate what is directly observed in papers (“observed” or bullet points of results) from what is inferred or generalized. One option is to label bullet lists under “Findings” vs “Commentary”. For instance:
  - *Finding (Evidence-backed):* “All three papers using Algorithm A report convergence within 10k iterations【Paper1†…, Paper2†…】.”
  - *Synthesis (Interpretation):* “We conclude Algorithm A scales linearly with data volume (since [findings]) but only under condition Z, as papers 4–6 note.”
  This echoes the practice in GRADE where you distinguish raw data from study-level confidence【29†L11-L14】.
- **Unresolved/Conflicting:** Clearly mark items that are disputed or have insufficient data. For example, if one group reports Method B works for real-time but another found it too slow, list both sides under a “Conflict” heading with their citations. This transparency avoids glossing over contradictions.
- **Confidence annotation:** Adopt a simple scale (e.g. High/Medium/Low) or qualitative labels for each synthesized conclusion. The scale should reflect quantity/quality of evidence (e.g. multiple experiments vs a single case study). In practice, even a note “(limited evidence)” is better than no cue. PRISMA-style systematic reviews also encourage stating evidence certainty【42†L235-L244】.
- **Bias and assumptions notes:** Where applicable, annotate potential biases (e.g. “all studies assume homogeneous servers” or “single-vendor platform”) and context limitations. This is part of epistemic discipline: don’t present any claim as universal if caveats exist.

In implementation, this could mean using a structured markup (Markdown with references, or XML tags) so the report is both human-readable and machine-parseable. Tables or formatted lists can include columns for *Claim*, *Source*, *Confidence*, and *Notes*. For example, a fragment might look like:

| Finding/Claim                          | Sources             | Confidence | Notes                   |
|----------------------------------------|---------------------|------------|-------------------------|
| “Algorithm A converges in <10k steps.” | [Paper1], [Paper2]  | High       | Measured on small dataset |
| “No studies report failures”           | (none)              | Low        | (Evidence gap)          |

Such tables embed traceability and illustrate uncertainty at a glance. In summary, Step 2 should treat all statements as evidence items with provenance, akin to PRISMA’s thorough reporting【42†L235-L244】, so that downstream engineers can trust and interpret the synthesis.

# Recommended synthesis methods / 建议的综合方法

Given a set of structured Step 1 records, the synthesis can proceed through multiple complementary methods:

- **Matrix and Tabulation:** Create cross-tabulation matrices where rows are concepts or attributes and columns are papers (or vice versa). This makes patterns explicit. For example, list all papers vs features like “supports encryption”, “requires GPU”, etc., ticking which papers mention them. This is a classic systematic mapping technique that highlights trends and gaps【3†L2824-L2831】【13†L78-L87】.
- **Taxonomies and Clustering:** Perform a taxonomy-based grouping of methods (e.g. by technique category, problem type, or component). This could be pre-defined (top-down) or derived (bottom-up via clustering similar abstracts/mechanisms). LLMs or topic models can aid here: Mallavarapu et al. show LLM-based topic modeling structuring review data into an explorable map【20†L128-L136】. Clustering papers by similarity can reveal “families” of solutions.
- **Contradiction/Conflict mapping:** As noted, explicitly list contradictions in a map or table. This could involve a binary “For/Against” chart for contested claims. No standard off-the-shelf algorithm exists, so this may be manual or semi-automated (e.g. flagging opposing keywords). The key is to catalog conflicts systematically rather than bury them in text.
- **Evidence mapping:** Use heatmaps or bubble charts to visualize where evidence is concentrated. For example, a matrix showing which research questions have been answered by many papers (high intensity) vs which have none (gaps). Khalil et al. suggest visual tools (heatmaps, networks) enhance SEM usability【13†L88-L93】. Similarly, you might produce a chart with method categories on one axis and evaluation metrics on another, with bubbles sized by number of studies.
- **Pattern cataloging:** Read through all extraction records to extract repeated themes or ‘rules of thumb’. This could be aided by coding techniques (like open coding in qualitative analysis) or using LLMs to suggest commonalities. The end result is a list of “patterns” (e.g. “Local caching of data reduces latency – noted in 3 papers”) that designers can reuse.
- **Assumption and risk mapping:** Besides content, one can use visual mappings (mind-map or force-directed graph) linking assumptions to papers, or even linking methods to risks. Though more experimental, such diagrams can clarify the space of underlying assumptions.

In practice, we recommend **iterative passes**: start with a broad tabular summary, then refine with targeted analyses. Automated tools (LLM classification, clustering libraries) can speed this up, but human oversight is key. For instance, start by having an LLM synthesize each extraction record into a short summary, then manually cluster those summaries into themes. Use tables (per 6.5.1 descriptive synthesis advice【3†L2824-L2831】) and consider network diagrams (nodes = papers or concepts) to visualize connections【13†L88-L93】. The combination of these methods helps ensure no single viewpoint dominates the synthesis.

# Recommended Step 2 report template options / 建议的步骤2报告模板选项

We consider several template styles:

- **Narrative review format:** Pro: very readable; Con: tends to be unstructured, with risk of hiding evidence in prose. Typical academic reviews use this, but as [42] warns, such reviews “往往没有标准化和明确的方法，结果有偏差”【42†L228-L236】. This format is too loose for our needs because it makes machine parsing and strict neutrality difficult.
- **Structured Markdown report:** Pro: good readability and moderate structure. Sections/headings clearly separate content (as we are using now). Lists, tables, and bullet points are easy to scan. References can be hyperlinked. Consistency must be enforced manually (e.g. style guides). This balances human and machine needs.
- **Layered report (Executive + Technical + Appendices):** Pro: addresses multiple audiences. The executive layer is narrative and high-level, while the technical layer is highly structured and data-rich. Appendices or embedded data files contain raw extraction records. This hybrid meets both readability and evidence traceability goals. Cons: slightly more complex to produce.
- **Machine-readable formats (XML/JSON):** Pro: fully parseable, supports automated tooling. We could tag each section/subsection for programmatic extraction, or output as JSON/YAML schema (like a config file). Con: far less human-friendly (needs tooling or transformation to read). This might be used as a **fallback** or internal representation behind the scenes. For example, an XML-tagged version of the report could facilitate automated QA or future updates, but its raw form is unwieldy for a human reader.
- **Mixed (Markdown with embedded metadata):** A middle ground is writing in Markdown but using embedded tags or front-matter (e.g., YAML in a markdown document) to encode structured fields. This allows both easy reading and automated extraction.

**Trade-offs:** A purely narrative report scores high on readability but low on traceability and neutrality. A pure XML report scores high on machine usability but low on human readability. The layered Markdown approach (our primary recommendation) offers a compromise: clear headings and prose for humans, plus tables and links for structure. It supports citations and can be lightly parsed (Markdown can be turned into HTML/JSON). For example, using Level-3 headings for each category (as above) and bullet tables for evidence will make scanning easy.

Our **primary template** suggestion is: a **strongly structured Markdown report** with fixed sections (as outlined in “Information architecture”), including embedded tables and lists where applicable. This can be validated by scripts (e.g. check that all section headers exist, all claims have citations). The **fallback** template could be an **XML/JSON hybrid**, where each section is a tag and each finding is an element. That would be used if full machine-parseability is needed (say to integrate with a design tool), but at the cost of human ergonomics.

# Recommended workflow for generating Step 2 / 建议的步骤2生成流程

We advocate a **multi-pass, iterative workflow** to build Step 2:

1. **Normalization pass:** Ingest all Step 1 records into a unified schema (common field names, data types). Use scripts or prompts to standardize terminology (e.g. merge synonyms, consistent units). This ensures consistency in what follows.
2. **Synthesis matrix pass:** Construct initial summary tables. For example, a matrix of “Paper × Key attributes” (as in previous section). Populate it from the normalized data. This quick view will already show recurring entries and blanks.
3. **Assumption/pattern identification pass:** Scan the tables and text for recurring themes. Use clustering or LLM-assisted grouping to collect similar assumptions or mechanisms. Create lists of assumptions, patterns, reusable components. This may involve prompting the model with subsets of records to extract common points.
4. **Contradiction pass:** Explicitly query for conflicts. For instance, prompt an LLM: “List any conflicts between these extraction records.” Or filter the matrix for cells with contradictory entries. Document them in a “Contradiction Map”.
5. **Production-readiness pass:** Assess practical viability. For each approach, note whether it’s been tested outside lab, what preconditions it has, etc. This may be partly automated (check record fields) and partly manual judgement.
6. **Synthesis writing pass:** Draft sections of the report. Use the structured outline to fill in narrative and tables. For example, prompt: “Summarize the recurring mechanisms from these records” or “Write a neutral overview of Approach X based on papers A, B, C.” Ensure to cite all sources.
7. **Review/validator pass:** Finally, run checks on the draft:
   - **Traceability check:** Are all factual statements cited? Every paragraph should have a reference footnote.
   - **Neutrality check:** Is any section unduly promoting one approach? A separate reviewer (human or LLM) should attempt to identify bias.
   - **Coverage check:** Did we include all papers and all major topics? A diff against the matrix can verify that no extraction record was ignored.
   - **Consistency check:** Ensure taxonomy, terminology, and confidence labels are used consistently.

This workflow is inspired by best practices in systematic mapping and evidence synthesis. For instance, Khalil et al. describe a “stepwise SEM workflow” involving coding and visualization【13†L58-L60】. Our steps adapt that to the engineering context. The key is to **iterate**: findings from later passes may feed back (e.g. a contradiction found may require revisiting the summary tables). Tools (scripts or LLM agents) can automate parts of this. But final validation should be done by experts to catch subtle issues.

# Prompting/model strategy / 提示和模型策略

Given the complexity, we recommend a *multi-stage prompting strategy* with GPT-5.4/Codex:

- **Structured prompts over monolithic ones:** Don’t attempt a single giant prompt to produce the whole report. Instead, break tasks into sub-prompts aligned with the workflow passes. For example:
  - *Data normalization:* feed in batches of Step 1 records and ask the model to output standardized JSON.
  - *Pattern extraction:* prompt with the combined normalized data, e.g. “What patterns recur across these records?”
  - *Table generation:* ask GPT to produce formatted markdown tables from a list of findings.
  - *Narrative writing:* provide context and bullet points, and ask for a paragraph summary.

- **Structured output tokens:** Guide the model to output in specific formats (e.g. JSON objects, Markdown tables) so that results can be parsed or validated. For instance, prompt: “Output a JSON list of {assumption:…, sources:[…]} entries.” Codex can generate code that organizes data well.

- **Chain-of-thought for high reasoning tasks:** Where we need inference or synthesis (e.g. evaluating evidence strength), allow GPT to “reason” step by step. Perhaps use few-shot examples. For high-level summarization (“Describe how approaches A and B differ”), a complex prompt with example outputs can help the model focus on differences.

- **Validator prompts:** After generation, we can use the model itself to critique the draft. For example: “Does the above summary show any bias towards a specific architecture? List unsupported claims.” This *reviewer pass* can catch issues that slip through.

- **Prompt templates:** Build reusable prompt templates for each pass, so the process is reproducible. For example, a template for “extract contradictions” that takes a text block of findings and returns conflict statements. Use placeholders for inserting different content.

Essentially, use a **multi-turn, structured prompting approach**. Let GPT-5.4 handle routine synthesis (high reasoning effort: summarization, pattern detection), but rely on custom validation and post-editing to ensure neutrality and completeness. Codex could be used to script the workflow itself (e.g. writing Python to organize the report content, if needed). Overall, the strategy is a staged pipeline: use tailored prompts for each subtask rather than one all-encompassing prompt. This approach mirrors the multi-pass workflow above and produces a more reliable, fine-grained result.

# Quality assurance framework / 质量保证框架

To ensure a Step 2 report is “good enough” for multiple designs, we propose a checklist and scoring rubric:

**Acceptance criteria / 检查要点:**
- **Completeness:** All Step 1 records are accounted for. The synthesis covers every topic identified in the extraction (verify via a table-of-contents or tagging).
- **Traceability:** Every factual claim or data point has a reference. Check that no paragraph lacks citations, and that the appendix crosslinks sources.
- **Neutrality:** The report contains no solution endorsement. Verify by searching for bias indicators (e.g. “best”, “must use”, or single-sided language). A design bias might show up as one approach repeatedly praised without caveat.
- **Clarity/Structure:** All recommended sections (taxonomy, assumptions, etc.) are present if relevant. The writing should be concise; long narrative blocks without headings are red flags.
- **Evidence vs. Opinion:** Opinions or interpretations should be labeled. If the report uses phrases like “it’s known that X works” without citation, that’s a flaw.
- **Uncertainty documented:** Any contradictory or uncertain point should be explicitly noted. Absence of any “unknowns” in a large corpus is suspicious.
- **Reusability hints:** The report should make clear how to branch into different design constraints (e.g. by labeling trade-offs such as cost vs performance). Lack of any discussion of design implications suggests it’s too generic.

**Scoring rubric:** Rate each criteria on, say, 1–5. For example:
- Coverage (all topics) 1–5,
- Citation rigor 1–5,
- Bias (low bias = high score) 1–5, etc.
A total score above a threshold means the report passes.

**Red flags:** Watch for language indicating design decisions (e.g. recommending a component without alternatives). Look for unsupported superlatives (“X clearly dominates”). If the synthesis reads like a marketing pitch, it fails. Narrative tangents not grounded in the cited literature are also red flags. If key sections (like the assumption map or evidence table) are completely missing, it suggests insufficient rigor.

For evaluation, one could set up a **benchmark** by running different Step 2 workflows on a common set of Step 1 data and comparing outputs against these criteria. This could involve peer review: have independent experts grade the reports for neutrality and completeness. Automated checks (e.g. “does each paragraph contain a reference pattern?”) could catch technical compliance issues. Over time, refine the checklist based on “lessons learned” from actual Step 3 outcomes (e.g. if designers had to re-run Step2, note what was missing).

# Recommended final methodology / 建议的最终方法论

Based on the above, the recommended Step 2 methodology is:

- **Preparation:** Establish a protocol (scope, schema) as you collected Step 1 records. Use this to guide synthesis (analogous to a review protocol).
- **Data organization:** Normalize and import all records into a unified format (spreadsheet or database). Conduct initial analysis (counts, keyword search) to familiarize with the corpus.
- **Synthesis passes:** Execute the multi-pass process: summarize each record, build matrices, extract themes/patterns, map contradictions and assumptions. Use a mix of automated (LLM, scripts) and manual methods.
- **Report assembly:** Draft the report following the structured template. Start with high-level sections (scope, methods), then fill in findings (taxonomy, patterns, assumptions, etc.) with supporting tables. Include an executive summary and a traceability appendix.
- **Quality checks:** Apply the QA framework above. Iterate until passing criteria. If time/budget allows, involve a second reviewer (or LLM verifier) to audit neutrality and coverage.
- **Output:** Produce the final report in the chosen format (e.g. Markdown). Also output any intermediate data (e.g. normalized record database, evidence maps) for future reference.

This methodology mirrors best practices in systematic mapping and evidence synthesis, tailored for engineering decision support. It is deliberately **design-neutral** and multi-purpose, enabling branching into conservative or innovative designs as needed.

# Minimum viable Step 2 template / 最简可行的步骤2模板

A minimal template might include:

- **Title and date**
- **Scope (1 paragraph):** What area is covered.
- **Corpus (list):** Brief list of included papers (title & ID).
- **Key Findings bullets (executive):** 3–5 bullet points of most important takeaways (with citations).
- **Summary Table:** One or two tables covering the most critical attributes (e.g. metrics, context) for all approaches.
- **Design Implications bullets:** 3–5 bullet points of actionable insights for designers (without saying “the design” is chosen).
- **References Appendix:** A raw list of source citations or a simple mapping of claims to sources.

This bare-bones version skips in-depth sections but delivers a quick synthesis for immediate use. It could be generated quickly by asking: “Given these extraction records, give me the 5 most important points and one comparative table.” It lacks advanced analysis (no detailed assumptions/contradictions sections), but satisfies basic reuse needs.

# Advanced Step 2 template / 高级步骤2模板

The advanced template builds on the above with full detail:

- **Cover page with version control** (authors, date, change log).
- **Table of contents** (links to sections).
- **Expanded Executive Summary:** Several paragraphs summarizing key results, uncertainties, and options.
- **Full Scope & Methodology:** Detailed explanation of corpus selection, synthesis procedure (even the list of prompts/methods used).
- **Detailed Taxonomy:** Possibly multiple levels of categories with subheadings.
- **Evidence Tables:** One or more extensive tables (or even spreadsheets) with rows for each finding/metric and columns for each paper.
- **Mechanism & Pattern Catalog:** Narrative plus small tables or diagrams listing each pattern, number of supporting papers, sources.
- **Assumption and Risk Maps:** Could include flowcharts or mapped diagrams showing how assumptions lead to risks.
- **Interactive elements:** If possible, include links to external visualization (e.g., an online evidence map) or embedded JSON for supporting tools.
- **Appendices:** Raw data appendix, including the full Step 1 records (or links to them), and any code or prompts used.
- **Glossary of terms** (optional) to clarify jargon.

This comprehensive template might be used when the stakes are high (e.g. designing a critical system) and there is budget for a thorough analysis. It essentially forms a living document that can be iteratively updated. It supports advanced queries: for instance, engineers could script queries against the JSON back-end or use the annotated document programmatically.

# Anti-patterns to avoid / 避免的反模式

Key anti-patterns include:

- **“Design-spray” reports:** Including specific architecture or product recommendations in Step 2.  E.g. saying “we will use Cloud Service X because it fits best” breaks neutrality.
- **Narrative fluff:** Writing long paragraphs without clear references or structure. As noted, narrative reviews tend to be biased and inconclusive【42†L228-L236】. Avoid storytelling style and prefer factual bullets and tables.
- **Cherry-picking:** Only summarizing papers that support an implicit thesis. We must not drop conflicting evidence to make a cleaner story.
- **Ambiguous claims:** Phrases like “most work assumes…” without citations are dangerous. Every statement of fact must trace to data.
- **Over-engineering the document format:** Going too far with XML or specialized schemas can make it hard for reviewers to read. Conversely, a completely unstructured dump of notes is useless. Balance is key.
- **Ignoring contradictions:** Skipping the contradictions section because it’s “messy” leads to a false sense of certainty.
- **Scope creep:** Letting Step 3 questions (e.g. “Should we use an event bus?”) drive Step 2. Step 2 should answer questions like “What has been done?” and “What do papers say about it?” – not “Which design should we pick?”

One can see parallels with known pitfalls in reviews: for instance, the medical review guide warns that missing key methodology details “impairs confidence in results”【42†L214-L223】. Similarly, a Step 2 report that omits methodology or evidence links is an anti-pattern. Avoid these by adhering strictly to the structured approach outlined above.

# Open questions / future improvements / 未解决问题与未来改进

Several areas invite further work:

- **Interactive synthesis tools:** Mallavarapu et al. show promise in “interactive evidence maps” that allow filtering and exploring evidence dynamically【20†L128-L136】. Building such a tool for engineering Step 2 (e.g. a web UI over the evidence tables) could greatly enhance usability.
- **Automation of assumption/contradiction detection:** Currently these still require human insight. Future AI models or NLP pipelines might automatically flag conflicting statements or extract tacit assumptions more reliably.
- **Standardized ontologies:** Developing a domain-specific ontology (or schema) for the kinds of findings in systems research could make Step 2 more consistent across projects.
- **Quality benchmarking:** Establish formal benchmarks (as hinted above) to evaluate Step 2 methods. This could become a field in itself: “meta-synthesis” of engineering literature.
- **Integration with Step 3 tools:** Ideally, Step 2 output could feed directly into design tools (e.g. performance modeling software). Investigating machine-readable interfaces (APIs) between reports and design environments is an open direction.
- **Crowdsourced updates:** As new papers arrive, how to update the synthesis without redoing everything? Developing *living* Step 2 reports with version control is a challenge.

Advances in automation, ML, and stakeholder engagement—as noted by SEM researchers—can refine these methods【13†L90-L94】. In summary, while the above methodology provides a solid foundation, we expect ongoing improvements in tooling and process will further streamline and enhance Step 2 over time.
