
import os
import sys
import json
import tempfile
import shutil
import logging

# Set environment variable for vLLM compatibility
os.environ['VLLM_USE_V1'] = '1'

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch

def run_demo():
    # Model path from user query
    model_path = "/data/Mamba/Data/hf_cache/hub/models--vandijklab--C2S-Scale-Gemma-2-2B/snapshots/7fc451a816ba12d47c85c5c5ad0036c994705d1f"
    
    print(f"Loading model from {model_path}...")

    # WORKAROUND: Disable softcapping to avoid flash attention compatibility issues
    # Read and modify model config to disable softcapping
    config_path = os.path.join(model_path, "config.json")
    
    temp_model_dir = None
    model_path_to_use = model_path

    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check if softcapping is enabled
        has_softcapping = config.get('attn_logit_softcapping') or config.get('final_logit_softcapping')
        
        if has_softcapping:
            print(f"Detected softcapping in model config. Creating temporary model directory with softcapping disabled...")
            
            # Create temporary directory
            temp_model_dir = tempfile.mkdtemp(prefix="vllm_model_")
            
            # Create symlinks for all files except config.json (much faster than copying)
            for item in os.listdir(model_path):
                if item != "config.json":
                    src = os.path.join(model_path, item)
                    dst = os.path.join(temp_model_dir, item)
                    if os.path.exists(dst):
                        os.remove(dst)
                    os.symlink(src, dst)
            
            # Write modified config
            config['attn_logit_softcapping'] = None
            config['final_logit_softcapping'] = None
            with open(os.path.join(temp_model_dir, "config.json"), 'w') as f:
                json.dump(config, f, indent=2)
            
            model_path_to_use = temp_model_dir
            print(f"Using temporary model directory: {temp_model_dir}")
    
    try:
        # Initialize vLLM
        llm = LLM(
            model=model_path_to_use,
            tensor_parallel_size=1,  # Assuming 1 GPU is available/enough for this demo
            gpu_memory_utilization=0.2,
            trust_remote_code=True,
            max_model_len=8192,
            enforce_eager=True # Sometimes helps with compatibility
        )
        
        tokenizer = llm.get_tokenizer()
        
        # --- Format 1 ---
        print("\n--- Preparing Format 1 ---")
        # Extract genes list from Format 2 (same genes as Format 2)
        # Format 2 uses comma-separated genes, Format 1 uses space-separated
        genes_list_format2 = "MALAT1, B2M, EEF1A1, RPL10, RPL41, RPL39, MT-CO2, RPS12, PTMA, ACTB, MT-CO1, RPL28, MT-CO3, RPL19, RPS15A, RPS28, RPS8, RPLP1, FTH1, IGLC3, RPL11, RPL34, RPS4X, TMSB4X, RPS13, RPL30, RPS14, RPS27, MT-CYB, RPS27A, RPS3, RPS23, RPL8, RPL18, RPL14, RPS19, CD74, RPL13, RPL32, MT-ATP6, TMSB10, RPL12, RPS10, RPL37, RPS3A, H3F3B, RPS2, RPL7A, HLA-B, TPT1, RPS26, FAU, RPL29, RPLP2, RPS24, RPL35A, RPL9, RPS18, RPL36, RPS25, IRF1, CD37, RPSA, NACA, RPL5, BTG1, IGHM, CD83, JUND, UBA52, RPL18A, RPS6, RPL13A, RACK1, RPL22, CDC42SE2, MAML2, EIF1, PFDN5, HLA-DRA, RPLP0, PSMB9, HLA-A, FTL, RPL24, ARHGAP15, RPL37A, HERPUD1, MT-ND4, MT-ND3, RPS15, HLA-E, CCSER1, ADAM28, RPL17, RPL6, RALGPS2, RPL26, HSP90AB1, RPL10A, RPL15, LINC01619, ZBTB20, RPS21, ARID1B, ATP5F1E, SKP1, SMCHD1, BAZ1A, SESN3, ZNF331, BACH2, GAPDH, RHOA, LINC-PINT, UBB, HLA-DPA1, ARHGAP24, LTB, HLA-DRB1, PNISR, NPM1, PDE3B, JAZF1, MAP3K8, ACTG1, SRP14, RPS16, RNF38, RPL3, RPS29, UBC, JMJD1C, BANK1, HNRNPDL, IFITM1, MIF, AFF3, RHOH, RPS7, HLA-C, SET, RPS5, DDX5, MBP, FAAH2, TNFAIP8, UCP2, HNRNPA2B1, CD247, OSBPL8, RPS11, HINT1, XRCC5, PTPRC, COX4I1, RPL7, SP100, PFN1, RPS4Y1, BTF3, LINC01578, COX7C, IL32, PABPC1, PITPNC1, RAC2, RPL23A, WIPF1, JAK1, EIF3K, PSMB1, H3F3A, NME2, MS4A1, NUCKS1, HAT1, MAP3K5, NT5C2, OAZ1, FAM49A, PTGES3, GGNBP2, TXNIP, CD96, TBC1D1, RPL27A, RPL35, HNRNPA3, PSME1, RSRP1, FOXP1, HSPA5, SH3BGRL3, ZNF644, CCT8, CFL1, TOMM7, SERP1, CNBP, ARHGDIB, ZFAND3, PAX5, TRBC2, EPC1, GRB2, FUS, XPO4, EBF1, GPCPD1, KAT6B, HMGB2, CALM2, SF1, VOPP1, DNAJC1, TAX1BP1, SEC62, SKIL, MED10, EEF1D, NEDD9, EXOC6B, MAX, BCL11B, UAP1, AGO4, NCOA3, GNAS, HNRNPC, SMYD3, ZNF721, ATP5MG, CSNK1A1, GPHN, PPP1R16B, SPPL3, CASP4, ARPC5, PAN3, PPP2R5C, RPL36AL, NAP1L1, HK1, KDM6A, SEPTIN7, ARID4B, PATJ, SSB, AKNA, ACTR3, MYL12A, ACTR2, PACS1, ELOB, OST4, KMT5B, SOS1, AUTS2, ITSN2, VIM, RSF1, ZEB1, LAMTOR1, BASP1, HSPA8, LRBA, TUBA1B, FOXN3, HSP90B1, SNU13, COMMD6, SRPK2, IGFLR1, EEF1B2, PDE4B, LSM14A, ERP29, HMBOX1, ELF1, TMEM135, STAT1, MITD1, SKAP1, TLK1, EPB41, NDUFV1, MGAT4A, COBLL1, INPP5A, BST2, ANP32B, NDUFV2, XAF1, ANKRD44, RPL36A, H1FX, RAPGEF6, GSTP1, IL4R, POU2F2, SAMD8, CCSER2, ST3GAL5, ANKRD11, TGFBR2, RNF130, RBX1, HNRNPU, PRKCB, ENSA, TBC1D5, BTBD11, HMGB1, DLEU2, ARPC3, ASAP1, UBXN1, CD52, HNRNPK, JUN, EXOC4, UQCR11, CIRBP, ERO1B, EIF1AY, ANKRD12, THOC7, HLA-DQB1, GBP1, RNPS1, ETV6, NOP10, RGS1, PEBP1, GPX4, CALR, KPNA6, DUT, RPL38, UQCRH, SUMO2, PFKFB3, CHST11, RPL21, PRMT2, KLF12, UBALD2, PRDM2, CD8B, CMSS1, TRMT112, ALOX5AP, C18orf32, ME2, RALA, HECTD1, LY9, ZNF37A, STK17A, USP34, FNBP1, SUB1, RAB18, MT-ND2, TSTD1, UBE2L3, DCXR, UTY, MED13L, MYL12B, NDUFB8, RABAC1, LY6E, TRPM7, XRN1, CD27, IFI6, TRBC1, TOMM20, LAT, CCNC, PARP14, ILKAP, RHOXF1, PALM2-AKAP2, YPEL5, CXCR4, TAB2, SAP18, CLIP4, EPC2, CRIP1, RPL4, RIC1, RBM6, SIAH1, CYLD, ASH1L, LPP, NUB1, FAM30A, ZHX2, LINC01138, YWHAH, NDUFA13, STK4, HNRNPD, ERH, VAMP2, PLEKHO1, SEC61B, NPEPPS, UBE2D3, CAMK4, RPL23, UBE2K, HADHA, TTN, RFC1, TIAL1, ABLIM1, TPD52, EIF4G3, SLC30A7, BRD4, ARHGEF18, STRBP, SON, UTRN, PHACTR2, RIF1, LLPH, UBE2V1, SNX9, EZR, MALT1, CDC14A, ITM2B, NIPSNAP2, PFDN2, COMMD10, YBX1, TCF4, ENO1, FIP1L1, SNHG29, SPIDR, GADD45B, GNG7, PPIG, OPTN, CYTIP, UQCRB, UBE2B, PSMD14, ATG14, U2AF1, MYL6, MTDH, ANXA6, G3BP1, SRSF5, EIF4A1, YWHAB, USP15, RBM38, ATP5PO, C5orf56, EIF2AK2, CWF19L1, ARGLU1, MSI2, ELF2, APPBP2, TMEM87A, COX6A1, ATP5F1A, CD48, ST6GAL1, SNRPB, TRAPPC9, S100A10, TMA7, JUNB, CNN2, PRKCA, PARP8, PPP1R7, PBRM1, FCHSD2, GMFG, LIMD2, PSMC5, BCAP31, UBAC2, FCRLA, TMEM161B, TMEM131L, PLP2"
        # Convert comma-separated to space-separated for Format 1
        genes_list_format1 = genes_list_format2.replace(", ", " ")
        
        format1_text = f"""Given the following cell sentence of 200 expressed genes representing a cell's basal state, predict the cell sentence after applying the perturbation: EIF4B.
Control cell sentence: {genes_list_format1}.

Perturbed cell sentence:"""
        
        # For Format 1, we treat it as a completion prompt since it ends with "Perturbed cell sentence:"
        # But if we want to follow the chat template fallback logic:
        if tokenizer.chat_template:
             # If we force it into chat template, it might look weird if we don't have a clear system/user split.
             # But Format 1 is presented as a single block.
             # Let's try completion first for Format 1 as it looks like one.
             prompt1 = format1_text
        else:
             prompt1 = format1_text

        # --- Format 2 ---
        print("\n--- Preparing Format 2 ---")
        system_msg = "You are a specialized AI assistant with expertise in single-cell genomics and cellular perturbation analysis. You can predict how gene expression profiles change after various perturbations such as drug treatments, genetic modifications, or environmental stimuli."
        user_msg = """How does B cell cells with drug perturbation: Edaravone reshape the gene expression from MALAT1, B2M, EEF1A1, RPL10, RPL41, RPL39, MT-CO2, RPS12, PTMA, ACTB, MT-CO1, RPL28, MT-CO3, RPL19, RPS15A, RPS28, RPS8, RPLP1, FTH1, IGLC3, RPL11, RPL34, RPS4X, TMSB4X, RPS13, RPL30, RPS14, RPS27, MT-CYB, RPS27A, RPS3, RPS23, RPL8, RPL18, RPL14, RPS19, CD74, RPL13, RPL32, MT-ATP6, TMSB10, RPL12, RPS10, RPL37, RPS3A, H3F3B, RPS2, RPL7A, HLA-B, TPT1, RPS26, FAU, RPL29, RPLP2, RPS24, RPL35A, RPL9, RPS18, RPL36, RPS25, IRF1, CD37, RPSA, NACA, RPL5, BTG1, IGHM, CD83, JUND, UBA52, RPL18A, RPS6, RPL13A, RACK1, RPL22, CDC42SE2, MAML2, EIF1, PFDN5, HLA-DRA, RPLP0, PSMB9, HLA-A, FTL, RPL24, ARHGAP15, RPL37A, HERPUD1, MT-ND4, MT-ND3, RPS15, HLA-E, CCSER1, ADAM28, RPL17, RPL6, RALGPS2, RPL26, HSP90AB1, RPL10A, RPL15, LINC01619, ZBTB20, RPS21, ARID1B, ATP5F1E, SKP1, SMCHD1, BAZ1A, SESN3, ZNF331, BACH2, GAPDH, RHOA, LINC-PINT, UBB, HLA-DPA1, ARHGAP24, LTB, HLA-DRB1, PNISR, NPM1, PDE3B, JAZF1, MAP3K8, ACTG1, SRP14, RPS16, RNF38, RPL3, RPS29, UBC, JMJD1C, BANK1, HNRNPDL, IFITM1, MIF, AFF3, RHOH, RPS7, HLA-C, SET, RPS5, DDX5, MBP, FAAH2, TNFAIP8, UCP2, HNRNPA2B1, CD247, OSBPL8, RPS11, HINT1, XRCC5, PTPRC, COX4I1, RPL7, SP100, PFN1, RPS4Y1, BTF3, LINC01578, COX7C, IL32, PABPC1, PITPNC1, RAC2, RPL23A, WIPF1, JAK1, EIF3K, PSMB1, H3F3A, NME2, MS4A1, NUCKS1, HAT1, MAP3K5, NT5C2, OAZ1, FAM49A, PTGES3, GGNBP2, TXNIP, CD96, TBC1D1, RPL27A, RPL35, HNRNPA3, PSME1, RSRP1, FOXP1, HSPA5, SH3BGRL3, ZNF644, CCT8, CFL1, TOMM7, SERP1, CNBP, ARHGDIB, ZFAND3, PAX5, TRBC2, EPC1, GRB2, FUS, XPO4, EBF1, GPCPD1, KAT6B, HMGB2, CALM2, SF1, VOPP1, DNAJC1, TAX1BP1, SEC62, SKIL, MED10, EEF1D, NEDD9, EXOC6B, MAX, BCL11B, UAP1, AGO4, NCOA3, GNAS, HNRNPC, SMYD3, ZNF721, ATP5MG, CSNK1A1, GPHN, PPP1R16B, SPPL3, CASP4, ARPC5, PAN3, PPP2R5C, RPL36AL, NAP1L1, HK1, KDM6A, SEPTIN7, ARID4B, PATJ, SSB, AKNA, ACTR3, MYL12A, ACTR2, PACS1, ELOB, OST4, KMT5B, SOS1, AUTS2, ITSN2, VIM, RSF1, ZEB1, LAMTOR1, BASP1, HSPA8, LRBA, TUBA1B, FOXN3, HSP90B1, SNU13, COMMD6, SRPK2, IGFLR1, EEF1B2, PDE4B, LSM14A, ERP29, HMBOX1, ELF1, TMEM135, STAT1, MITD1, SKAP1, TLK1, EPB41, NDUFV1, MGAT4A, COBLL1, INPP5A, BST2, ANP32B, NDUFV2, XAF1, ANKRD44, RPL36A, H1FX, RAPGEF6, GSTP1, IL4R, POU2F2, SAMD8, CCSER2, ST3GAL5, ANKRD11, TGFBR2, RNF130, RBX1, HNRNPU, PRKCB, ENSA, TBC1D5, BTBD11, HMGB1, DLEU2, ARPC3, ASAP1, UBXN1, CD52, HNRNPK, JUN, EXOC4, UQCR11, CIRBP, ERO1B, EIF1AY, ANKRD12, THOC7, HLA-DQB1, GBP1, RNPS1, ETV6, NOP10, RGS1, PEBP1, GPX4, CALR, KPNA6, DUT, RPL38, UQCRH, SUMO2, PFKFB3, CHST11, RPL21, PRMT2, KLF12, UBALD2, PRDM2, CD8B, CMSS1, TRMT112, ALOX5AP, C18orf32, ME2, RALA, HECTD1, LY9, ZNF37A, STK17A, USP34, FNBP1, SUB1, RAB18, MT-ND2, TSTD1, UBE2L3, DCXR, UTY, MED13L, MYL12B, NDUFB8, RABAC1, LY6E, TRPM7, XRN1, CD27, IFI6, TRBC1, TOMM20, LAT, CCNC, PARP14, ILKAP, RHOXF1, PALM2-AKAP2, YPEL5, CXCR4, TAB2, SAP18, CLIP4, EPC2, CRIP1, RPL4, RIC1, RBM6, SIAH1, CYLD, ASH1L, LPP, NUB1, FAM30A, ZHX2, LINC01138, YWHAH, NDUFA13, STK4, HNRNPD, ERH, VAMP2, PLEKHO1, SEC61B, NPEPPS, UBE2D3, CAMK4, RPL23, UBE2K, HADHA, TTN, RFC1, TIAL1, ABLIM1, TPD52, EIF4G3, SLC30A7, BRD4, ARHGEF18, STRBP, SON, UTRN, PHACTR2, RIF1, LLPH, UBE2V1, SNX9, EZR, MALT1, CDC14A, ITM2B, NIPSNAP2, PFDN2, COMMD10, YBX1, TCF4, ENO1, FIP1L1, SNHG29, SPIDR, GADD45B, GNG7, PPIG, OPTN, CYTIP, UQCRB, UBE2B, PSMD14, ATG14, U2AF1, MYL6, MTDH, ANXA6, G3BP1, SRSF5, EIF4A1, YWHAB, USP15, RBM38, ATP5PO, C5orf56, EIF2AK2, CWF19L1, ARGLU1, MSI2, ELF2, APPBP2, TMEM87A, COX6A1, ATP5F1A, CD48, ST6GAL1, SNRPB, TRAPPC9, S100A10, TMA7, JUNB, CNN2, PRKCA, PARP8, PPP1R7, PBRM1, FCHSD2, GMFG, LIMD2, PSMC5, BCAP31, UBAC2, FCRLA, TMEM161B, TMEM131L, PLP2? List the top resulting genes.

Perturbation details: Edaravone primarily functions as a potent free radical scavenger by directly targeting and neutralizing reactive oxygen species (ROS), such as hydroxyl and peroxyl radicals, thereby preventing oxidative damage to lipids, proteins, and DNA. This reduction in oxidative stress subsequently modulates the expression of key cytoprotective genes, typically upregulating antioxidant response elements like *HMOX1* and *NQO1* via the NRF2/KEAP1 pathway, while downregulating pro-inflammatory genes such as *TNF* and *IL6* through inhibition of the NF-κB signaling cascade. Consequently, these transcriptional changes lead to the stabilization of mitochondrial function, a decrease in apoptotic signaling, and an overall attenuation of neuroinflammatory pathways, which collectively promote neuronal survival in conditions like ALS and ischemic stroke."""
        
        # Ground truth for Format 2 (from user's example)
        format2_ground_truth = "Gene abundance will be led by: MALAT1, IGKC, MT-CO2, B2M, MT-CO1, MT-ATP6, CD74, RPL39, TMSB4X, RPLP1, MT-CO3, RPS8, HLA-DRA, PTMA, MT-CYB, RPL10, RPS27A, EEF1A1, RPS12, RPL41, RPL30, RPL9, RPS13, RPS23, TMSB10, LINC-PINT, UHRF2, RPL13, RPL8, MT-ND4, RPL34, JUND, RPS3, HLA-B, SIPA1L1, ADK, RPL28, JARID2, SRPK2, RPS4X, RPLP2, PALM2-AKAP2, AFF3, RPL37, FAU, CD37, ARID5B, CD83, BCL2, RPL12, FAM107B, ZCCHC7, RPS26, RPL32, RPS24, RPL11, BPTF, HLA-DRB1, DDX5, RPS2, IGHM, RPL26, RPL29, RPS28, RPS16, SIPA1L3, EPS15, RPS15, NR3C1, RPL18, UBA52, CLEC2D, TPT1, PDE3B, CIRBP, PRDM2, SP100, SIK3, RPL18A, FOXP1, HLA-DQA1, HLA-DQB1, LPP, ATP8A1, RPS14, FTL, RPL19, BLK, TNRC6B, DIAPH2, RERE, RPS27, RPL14, TBC1D5, ARPC3, RPS15A, HLA-E, HLA-A, RABGAP1L, LINC01578, LYN, RPL15, MBNL1, RPS10, AC097376.3, RPS18, DOCK8, AC120193.1, SNHG29, HLA-DPB1, ELMO1, RPS3A, NEAT1, ACTB, MARCH3, OAZ1, EHMT1, FCHSD2, SGO1-AS1, FTH1, UBE2E2, ATP2B1, AL139020.1, RPL36, SNX9, UBB, H3F3B, SMCHD1, EXOC4, RPL5, CCR7, MT-ND5, RALGPS2, EEF1D, BTG1, RPS25, HNRNPA1, PCDH9, CAMK1D, TCL1A, MYCBP2, ARHGAP15, STRBP, ATP5IF1, HSP90AB1, GMDS-DT, MEF2C, HLA-C, SEPTIN7, PTPN2, GRB2, AIG1, SLAIN2, TNIK, HNRNPD, ADAM28, RABEP1, CYLD, GPATCH8, DDX17, RPL7, AKAP13, CASK, MT-ND3, LUC7L2, TRBC2, FAM3C, IMMP2L, MEF2A, SLC25A17, ARID1B, MAML3, HNRNPA2B1, STEAP1B, MED23, HLA-DRB5, WHAMM, KMT2E, RPL24, GOLGB1, ARIH1, PRKCE, PLEKHA2, PTK2, ZHX2, S1PR1, RSRP1, SYNE2, KAT6B, TGFBR2, INPP5F, PHF21A, RPL7A, RPL22, CXCR4, PELI1, ZEB2, PBX3, RPL35, PSIP1, MICU2, HMGB1, CD48, NIN, CHURC1, EAPP, RPS6KA5, NUP214, HMGA1P4, CLTA, TAOK3, CCDC91, PCGF5, CUL5, STAT1, PSMD4, CHD6, RPS5, TMEM131L, RPRD2, HIBCH, RPS9, RELB, ZFP64, SP110, RNF115, LCOR, PTPN4, RPS21, LMBRD1, RPL35A, CYB5R4, GNAS, FANCB, GLCCI1, FBXO28, YWHAB, RICTOR, DIAPH1, SSBP2, STK4, MAGT1, WDR73, NLRC5, ATL2, CTDSPL2, DFFA, VPS13C, CCPG1, OGT, ANKRD11, NUP88, SLTM, RNF111, LINC00926, PRKCB, METTL9, MECP2, CAMKMT, PPM1B, CHCHD3, IQCB1, ATG3, PAK2, SEPSECS, HERC1, PARN, MYO1E, GSAP, MDH2, LINC01004, TNKS, PPP1R3B, CHD2, RNF2, LITAF, LUC7L3, MAPRE2, LDLRAD4, NDUFV2, SELENOT, SPAG16, MICU1, ANK3, THAP6, SCAF8, ZDHHC14, SRSF11, TYW1, EIF1, APPBP2, SUMO2, PPIA, AUTS2, NIPSNAP2, VOPP1, TOMM7, SFPQ, TTC14, SDAD1, TMEM156, STAP1, SCFD2, RHOH, UBE2K, TPST1, UTRN, CCM2, TUT4, MAN2A1, EIF3M, NUP160, STK17B, ATG7, TLE1, TLE4, CD52, API5, CD44, UVRAG, VPS51, TNFAIP8, DMXL1, PAIP2, EIF4G3, ILKAP, DGKD, INPP5D, EEF1B2, HDLBP, GPR155, CAB39, LINC00623, NAP1L4, VPS26A, EDRF1, FAM53B, AP001636.3, SF1, TPI1, FAM117B, ORC2, KMT2A, CASP4, DAZAP2, RPL36AL, RPS29, MTA3, TMBIM4, USP15, MYL6, RAP1B, IFNG-AS1, TRIM22, ABLIM1, BTRC, MED13L, RPL6, FCRL3, SH3BP5, RBSN, OVCH1-AS1, TBC1D15, CNOT2, HIPK1, MLLT3, GAPVD1, ERP44, CNTRL, HVCN1, SRSF9, CHPT1, SETX, ANKRD13A, DRAM1, NBEA, FOXO1, RFX3, RAB11FIP1, KIAA2026, PAPOLG, IWS1, GYPC, FBRSL1, TAGLN2, ZNF33A, CELF2, EPC1, RNASEH2B, TPT1-AS1, GNG2, DLEU2, ITM2B, KCMF1, VRK2, AFTPH, UBXN2A, HEATR5B, DGUOK, COX5B, TMCO4, UBC, AL592295.3, WDFY2, MIS18BP1, RCSD1, IRF9, PSME1, FOXN3, THRAP3, CUX1, HBP1, GNB5, NDUFA1, KDM5D, RPL36A, RBM6, STRN3, NGDN, PABPN1, KPNA1, XRN1, SPIDR, NSD3, PDE7A, ZBTB1, ELMSAN1, SELL, COX6C, TMEM71, SECISBP2, YWHAZ, DRAM2, KIAA1143, MRPL3, SEC22A, LTB, SESN1, AHI1, TRMT11, ADD1, PSMF1, ZNF581, PHLPP2, DHRSX, SNU13, PIGN, PWWP3A, C18orf32, STIM2, LRBA, JUN, PTPRC, BRD7, LONP2, RAC2, ITCH, SAFB, ATP5F1E, TTC3, BTG2, COX4I1, AC018695.9, CMIP, PSMB3, CYBB, RPS4Y1, BRCC3, ZNF737, GALNT2, RPL13A, NCOA3, PDE4D, SNX13, CDK13, SYPL1, ANKHD1, GPBP1, MAP3K1, NASP, PPIP5K2, MBP, LMAN1, JUNB, TPK1, SLCO5A1, BRAF, E2F5, LUC7L, EPB41L4A-AS1, MCCC2, BTF3, KDM6A, HUWE1, RPS6KA3, PLP2, RPGR, ABCB4, CCDC18-AS1"
        
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        
        if tokenizer.chat_template:
            prompt2 = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            print("No chat template found, using fallback format...")
            prompt2 = f"{system_msg}\n\nUser: {user_msg}\n\nAssistant:"
        
        prompts = [prompt1, prompt2]
        ground_truths = [None, format2_ground_truth]  # Format 1 has no ground truth, Format 2 has ground truth
        
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=1000,
            stop_token_ids=[tokenizer.eos_token_id] if hasattr(tokenizer, 'eos_token_id') else None
        )
        
        print("\n--- Running Inference ---")
        outputs = llm.generate(prompts, sampling_params)
        
        print("\n\n=== RESULTS ===\n")
        
        print("=" * 80)
        print("--- FORMAT 1 RESULT ---")
        print("=" * 80)
        print(f"\nPrompt:\n{prompts[0]}...\n")
        print(f"Predicted Output:\n{outputs[0].outputs[0].text}\n")
        print(f"Ground Truth:\nN/A (not provided)\n")
        print("-" * 80)
        
        print("\n" + "=" * 80)
        print("--- FORMAT 2 RESULT ---")
        print("=" * 80)
        print(f"\nPrompt:\n{prompts[1]}...\n")
        print(f"Predicted Output:\n{outputs[1].outputs[0].text}\n")
        print(f"Ground Truth:\n{ground_truths[1]}\n")
        print("-" * 80)
        print()

    finally:
        # Clean up temporary directory if created
        if temp_model_dir and os.path.exists(temp_model_dir):
            print(f"Cleaning up temporary model directory: {temp_model_dir}")
            try:
                shutil.rmtree(temp_model_dir)
            except Exception as e:
                print(f"Failed to clean up temporary directory: {e}")

if __name__ == "__main__":
    run_demo()
