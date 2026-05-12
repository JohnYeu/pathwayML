# PathwayML-Ath V6 reference and accessibility audit

Each cited work was checked for a DOI, official article page, open PDF, PubMed Central record, arXiv record, or official database/resource page. The paper reference list includes a DOI or open/readable path for every item.

1. Aleksander, S. A., Balhoff, J., Carbon, S., Cherry, J. M., Drabkin, H. J., Ebert, D., Feuermann, M., et al. 2023. The Gene Ontology Knowledgebase in 2023. Genetics 224(1): iyad031. DOI: 10.1093/genetics/iyad031. Open/readable path: https://academic.oup.com/genetics/article/224/1/iyad031/7068118

2. Berardini, T. Z., Reiser, L., Li, D., Mezheritsky, Y., Muller, R., Strait, E., and Huala, E. 2015. The Arabidopsis Information Resource: making and mining the gold standard annotated reference plant genome. Genesis 53(8): 474-485. DOI: 10.1002/dvg.22877. Open/readable path: https://phoenixbioinformatics.atlassian.net/wiki/spaces/COM/pages/42216383

3. Breiman, L. 2001. Random Forests. Machine Learning 45: 5-32. DOI: 10.1023/A:1010933404324. Open/readable path: https://www.stat.berkeley.edu/~breiman/randomforest2001.pdf

4. Chen, T., and Guestrin, C. 2016. XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794. DOI: 10.1145/2939672.2939785. Open/readable path: https://arxiv.org/abs/1603.02754

5. Dixon, D. P., and Edwards, R. 2010. Glutathione transferases. The Arabidopsis Book 8: e0131. DOI: 10.1199/tab.0131. Open/readable path: https://pmc.ncbi.nlm.nih.gov/articles/PMC3244946/

6. Du, J., Jia, P., Dai, Y., Tao, C., Zhao, Z., and Zhi, D. 2019. Gene2vec: distributed representation of genes based on co-expression. BMC Genomics 20(Suppl 1): 82. DOI: 10.1186/s12864-018-5370-x. Open/readable path: https://pmc.ncbi.nlm.nih.gov/articles/PMC6360648/

7. Hawkins, C., Ginzburg, D., Zhao, K., Dwyer, W., Xue, B., Xu, A., Rice, S., et al. 2025. Plant Metabolic Network 16: expansion of metabolic pathway databases and resources. Nucleic Acids Research 53(D1): D1606-D1613. DOI: 10.1093/nar/gkae991. Open/readable path: https://academic.oup.com/nar/article/53/D1/D1606/7876800

8. Kanehisa, M., Sato, Y., and Kawashima, M. 2023. KEGG for taxonomy-based analysis of pathways and genomes. Nucleic Acids Research 51(D1): D587-D592. DOI: 10.1093/nar/gkac963. Open/readable path: https://academic.oup.com/nar/article/51/D1/D587/6775388

9. Langfelder, P., and Horvath, S. 2008. WGCNA: an R package for weighted correlation network analysis. BMC Bioinformatics 9: 559. DOI: 10.1186/1471-2105-9-559. Open/readable path: https://pmc.ncbi.nlm.nih.gov/articles/PMC2631488/

10. Lundberg, S. M., and Lee, S.-I. 2017. A unified approach to interpreting model predictions. Advances in Neural Information Processing Systems 30. Open/readable path: https://arxiv.org/abs/1705.07874

11. McInnes, L., Healy, J., Saul, N., and Grossberger, L. 2018. UMAP: Uniform Manifold Approximation and Projection. Journal of Open Source Software 3(29): 861. DOI: 10.21105/joss.00861. Open/readable path: https://joss.theoj.org/papers/10.21105/joss.00861

12. Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., et al. 2011. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research 12: 2825-2830. Open/readable path: https://www.jmlr.org/papers/volume12/pedregosa11a/pedregosa11a.pdf

13. Rives, A., Meier, J., Sercu, T., Goyal, S., Lin, Z., Liu, J., Guo, D., et al. 2021. Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences. Proceedings of the National Academy of Sciences 118(15): e2016239118. DOI: 10.1073/pnas.2016239118. Open/readable path: https://www.pnas.org/doi/10.1073/pnas.2016239118

14. Subramanian, A., Tamayo, P., Mootha, V. K., Mukherjee, S., Ebert, B. L., Gillette, M. A., Paulovich, A., et al. 2005. Gene set enrichment analysis: a knowledge-based approach for interpreting genome-wide expression profiles. Proceedings of the National Academy of Sciences 102(43): 15545-15550. DOI: 10.1073/pnas.0506580102. Open/readable path: https://www.pnas.org/doi/10.1073/pnas.0506580102

## Data-source access checks

- KEGG REST API: official REST-style API at https://www.kegg.jp/kegg/api.html; pathway links can be read through https://rest.kegg.jp.
- GO annotations: official Gene Ontology annotation downloads include Arabidopsis thaliana / TAIR / tair.gaf at https://current.geneontology.org/products/pages/downloads.html.
- PlantCyc/AraCyc/PMN: PMN data-download page states that complete databases including AraCyc and PlantCyc can be downloaded after a freely available licence agreement: https://plantcyc.org/data-downloads/.
- TAIR: official Arabidopsis resource page at https://www.arabidopsis.org/.

## Removed or avoided claims/references

- No placeholder GitHub repository is cited as an active public repository. The code is provided as an accompanying archive and a public URL should be added before external submission.
- The C1 stress/glutathione set is not described as a novel pathway because ORA/overlap analysis indicates known glutathione-metabolism association.
- Unsupported numerical estimates such as an unmeasured co-annotation-only AUROC range and an unverified percentage of genes lacking GO annotation were removed.
