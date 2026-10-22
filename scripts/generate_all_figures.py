import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

# ==============================================================================
# 1. BUILD DRAW.IO XML FOR FIGURE 1: SENSING-TO-INFERENCE WORKFLOW
# ==============================================================================
def build_fig1_drawio_xml():
    root = ET.Element('mxfile', host='app.diagrams.net', modified='2026-10-07T08:00:00.000Z', agent='Antigravity', version='24.0.0', type='device')
    diagram = ET.SubElement(root, 'diagram', id='fig1_workflow', name='Figure 1 - Sensing-to-Inference Workflow')
    model = ET.SubElement(diagram, 'mxGraphModel', dx='1600', dy='900', grid='1', gridSize='10', guides='1', tooltips='1', connect='1', arrows='1', fold='1', page='1', pageScale='1', pageWidth='1600', pageHeight='700', background='#FFFFFF', math='1', shadow='0')
    root_cell = ET.SubElement(model, 'root')
    ET.SubElement(root_cell, 'mxCell', id='0')
    ET.SubElement(root_cell, 'mxCell', id='1', parent='0')

    def add_cell(id_str, val, style, x, y, w, h, parent='1', vertex='1', edge=None, source=None, target=None):
        attribs = {'id': str(id_str), 'parent': str(parent)}
        if vertex: attribs['vertex'] = str(vertex)
        if edge: attribs['edge'] = str(edge)
        if source: attribs['source'] = str(source)
        if target: attribs['target'] = str(target)
        attribs['value'] = val
        attribs['style'] = style
        c = ET.SubElement(root_cell, 'mxCell', **attribs)
        geom = ET.SubElement(c, 'mxGeometry')
        if edge:
            geom.set('relative', '1')
            geom.set('as', 'geometry')
        else:
            geom.set('x', str(x))
            geom.set('y', str(y))
            geom.set('width', str(w))
            geom.set('height', str(h))
            geom.set('as', 'geometry')
        return str(id_str)

    title_style = "text;html=1;align=center;verticalAlign=middle;fontSize=18;fontStyle=1;fontFamily=Helvetica;fontColor=#1A2530;"
    add_cell(2, "End-to-End Spaceborne Sensing-to-Inference Framework for XCO2 Plume Delineation", title_style, 250, 15, 1100, 30)

    # 5 Major Stage Containers
    c_stage1 = "swimlane;startSize=28;horizontal=1;container=1;collapsible=0;rounded=1;arcSize=6;strokeWidth=1.5;fillColor=#F4F7FB;strokeColor=#2B6CB0;fontFamily=Helvetica;fontSize=11;fontStyle=1;align=center;fontColor=#1D3557;"
    c_stage2 = "swimlane;startSize=28;horizontal=1;container=1;collapsible=0;rounded=1;arcSize=6;strokeWidth=1.5;fillColor=#FFF8F0;strokeColor=#C05621;fontFamily=Helvetica;fontSize=11;fontStyle=1;align=center;fontColor=#7B341E;"
    c_stage3 = "swimlane;startSize=28;horizontal=1;container=1;collapsible=0;rounded=1;arcSize=6;strokeWidth=1.5;fillColor=#F2F9F5;strokeColor=#276749;fontFamily=Helvetica;fontSize=11;fontStyle=1;align=center;fontColor=#1B4332;"
    c_stage4 = "swimlane;startSize=28;horizontal=1;container=1;collapsible=0;rounded=1;arcSize=6;strokeWidth=1.5;fillColor=#FAF5FF;strokeColor=#6B46C1;fontFamily=Helvetica;fontSize=11;fontStyle=1;align=center;fontColor=#322659;"
    c_stage5 = "swimlane;startSize=28;horizontal=1;container=1;collapsible=0;rounded=1;arcSize=6;strokeWidth=1.5;fillColor=#FEFCBF;strokeColor=#B7791F;fontFamily=Helvetica;fontSize=11;fontStyle=1;align=center;fontColor=#744210;"

    b_sub = "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#CBD5E0;strokeWidth=1.2;fontFamily=Helvetica;fontSize=10;align=center;fontColor=#2D3748;"
    b_core = "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#EDF2F7;strokeColor=#4A5568;strokeWidth=1.5;fontFamily=Helvetica;fontSize=10;fontStyle=1;align=center;fontColor=#1A202C;"
    edge_flow = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#2B6CB0;strokeWidth=2.2;endArrow=classic;endFill=1;"

    # Stage 1: Spaceborne GHG Observation
    add_cell(10, "1. Spaceborne Remote Sensing", c_stage1, 30, 60, 270, 480)
    add_cell(11, "&lt;b&gt;Satellite Platform&lt;/b&gt;&lt;br&gt;CO2M / OCO-2 / OCO-3&lt;br&gt;Polar Sun-Synchronous Orbit&lt;br&gt;Altitude ~705 km", b_sub, 50, 100, 230, 70)
    add_cell(12, "&lt;b&gt;SWIR Spectrometer&lt;/b&gt;&lt;br&gt;CO2 absorption bands (1.6 &amp; 2.06 µm)&lt;br&gt;Spatial resolution 2×2 km²&lt;br&gt;Push-broom imaging swath", b_sub, 50, 190, 230, 80)
    add_cell(13, "&lt;b&gt;Atmospheric Retrieval&lt;/b&gt;&lt;br&gt;Full-physics radiative transfer&lt;br&gt;XCO2 column-averaged dry-air mole fraction (ppm)", b_sub, 50, 290, 230, 75)
    add_cell(14, "&lt;b&gt;Target Scenes&lt;/b&gt;&lt;br&gt;Urban areas (Berlin, Paris)&lt;br&gt;Power plants (Boxberg, Lippendorf)", b_core, 50, 385, 230, 60)

    # Stage 2: Observation & Noise Artifacts
    add_cell(20, "2. Scene Tile Extraction &amp; Noise", c_stage2, 330, 60, 270, 480)
    add_cell(21, "&lt;b&gt;Regional Tile Cutout&lt;/b&gt;&lt;br&gt;Grid size: H × W (e.g. 64 × 64)&lt;br&gt;Centered over emission source", b_sub, 350, 100, 230, 65)
    add_cell(22, "&lt;b&gt;Atmospheric Background&lt;/b&gt;&lt;br&gt;Large-scale biogenic gradients&lt;br&gt;Mean baseline ~402–415 ppm&lt;br&gt;Spatial trends &amp; surface albedo", b_sub, 350, 185, 230, 75)
    add_cell(23, "&lt;b&gt;Observational Degradation&lt;/b&gt;&lt;br&gt;• Random instrument noise (σ ~ 0.7 ppm)&lt;br&gt;• Cloud &amp; aerosol screening artifacts&lt;br&gt;• Weak anomaly contrast (ΔXCO2 ~ 1–3 ppm)", b_core, 350, 280, 230, 95)
    add_cell(24, "&lt;b&gt;Noisy Observation Tile&lt;/b&gt;&lt;br&gt;&lt;b&gt;X&lt;/b&gt; ∈ ℝ^(B × 1 × H × W)", b_sub, 350, 395, 230, 50)

    # Stage 3: Proposed Joint Restoration-Aware Engine
    add_cell(30, "3. Proposed Joint Analytics Engine", c_stage3, 630, 60, 330, 480)
    add_cell(31, "&lt;b&gt;Shared ConvTransformer Encoder&lt;/b&gt;&lt;br&gt;Multi-scale hierarchical tokens (F1, F2, F3)&lt;br&gt;Local-window attention + Relative pos bias", b_core, 650, 100, 290, 70)
    add_cell(32, "&lt;b&gt;Auxiliary Denoising Pathway&lt;/b&gt;&lt;br&gt;Pretrained reconstruction decoder&lt;br&gt;Total variation &amp; Plume mass conservation&lt;br&gt;Continuous concentration regression", b_sub, 650, 190, 290, 80)
    add_cell(33, "&lt;b&gt;Multi-Scale Segmentation Pathway&lt;/b&gt;&lt;br&gt;• Feature Pyramid Network (FPN)&lt;br&gt;• Channel-gating recalibration (GAP+Sigmoid)&lt;br&gt;• Multi-resolution fusion &amp; ConvTransformer head", b_sub, 650, 290, 290, 85)
    add_cell(34, "&lt;b&gt;Physics Loss Coupling&lt;/b&gt;&lt;br&gt;L_total = 0.3 L_den + 0.7 L_seg&lt;br&gt;L_phys penalty for baseline violations", b_core, 650, 395, 290, 65)

    # Stage 4: Calibrated Quantitative Outputs
    add_cell(40, "4. Quantitative Outputs", c_stage4, 990, 60, 270, 480)
    add_cell(41, "&lt;b&gt;Restored XCO2 Field (X̂)&lt;/b&gt;&lt;br&gt;Denoised continuous concentrations&lt;br&gt;Subtle plume enhancements preserved&lt;br&gt;Noise floor suppressed", b_core, 1010, 110, 230, 80)
    add_cell(42, "&lt;b&gt;Plume Probability Map (M̂)&lt;/b&gt;&lt;br&gt;Per-pixel plume confidence ∈ [0, 1]&lt;br&gt;Calibrated spatial uncertainty&lt;br&gt;Sharp plume boundaries", b_core, 1010, 220, 230, 80)
    add_cell(43, "&lt;b&gt;Binary Mask Delineation&lt;/b&gt;&lt;br&gt;Thresholded footprint M_thresh&lt;br&gt;Isolated point/urban plume geometry&lt;br&gt;High IoU and low NWBCE", b_sub, 1010, 330, 230, 80)

    # Stage 5: Downstream Carbon Analytics & Applications
    add_cell(50, "5. Geospatial &amp; Climate Applications", c_stage5, 1290, 60, 280, 480)
    add_cell(51, "&lt;b&gt;Cross-Sectional Flux Inversion&lt;/b&gt;&lt;br&gt;Integration with wind speed vectors:&lt;br&gt;Q = ∫ U_eff · (XCO2 - X_bg) dl", b_sub, 1310, 100, 240, 75)
    add_cell(52, "&lt;b&gt;Point Source Verification&lt;/b&gt;&lt;br&gt;Facility-level reporting vs. inventories&lt;br&gt;Independent MRV (Measurement, Reporting &amp; Verification)", b_sub, 1310, 195, 240, 75)
    add_cell(53, "&lt;b&gt;Urban Carbon Budgets&lt;/b&gt;&lt;br&gt;Distinguishing diffuse urban emissions&lt;br&gt;from surrounding biosphere fluxes", b_sub, 1310, 290, 240, 75)
    add_cell(54, "&lt;b&gt;GIS &amp; Policy Decision Support&lt;/b&gt;&lt;br&gt;Operational climate mitigation actions&lt;br&gt;Global Stocktake compliance", b_core, 1310, 385, 240, 65)

    # Inter-stage arrows
    add_cell(101, "", edge_flow, 0, 0, 0, 0, edge=1, source=10, target=20)
    add_cell(102, "", edge_flow, 0, 0, 0, 0, edge=1, source=20, target=30)
    add_cell(103, "", edge_flow, 0, 0, 0, 0, edge=1, source=30, target=40)
    add_cell(104, "", edge_flow, 0, 0, 0, 0, edge=1, source=40, target=50)

    # Bottom workflow summary
    b_legend = "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#F8F9FA;strokeColor=#BDC3C7;strokeWidth=1;fontFamily=Helvetica;fontSize=10;fontColor=#2D3748;align=center;"
    add_cell(60, "&lt;b&gt;Key Pipeline Advantage:&lt;/b&gt; By coupling satellite restoration and segmentation in a shared hierarchical representation, the framework overcomes low signal-to-noise ratios (SNR), preserves weak diffuse plume tails, and provides reliable inputs for downstream carbon flux inversion.", b_legend, 30, 560, 1540, 35)

    xml_str = minidom.parseString(ET.tostring(root, encoding='utf-8')).toprettyxml(indent="  ")
    return xml_str

# ==============================================================================
# 2. BUILD DRAW.IO XML FOR FIGURE 3: CONVTRANSFORMER BLOCK
# ==============================================================================
def build_fig3_drawio_xml():
    root = ET.Element('mxfile', host='app.diagrams.net', modified='2026-10-07T08:00:00.000Z', agent='Antigravity', version='24.0.0', type='device')
    diagram = ET.SubElement(root, 'diagram', id='fig3_convtransformer', name='Figure 3 - ConvTransformer Block')
    model = ET.SubElement(diagram, 'mxGraphModel', dx='1400', dy='900', grid='1', gridSize='10', guides='1', tooltips='1', connect='1', arrows='1', fold='1', page='1', pageScale='1', pageWidth='1400', pageHeight='750', background='#FFFFFF', math='1', shadow='0')
    root_cell = ET.SubElement(model, 'root')
    ET.SubElement(root_cell, 'mxCell', id='0')
    ET.SubElement(root_cell, 'mxCell', id='1', parent='0')

    def add_cell(id_str, val, style, x, y, w, h, parent='1', vertex='1', edge=None, source=None, target=None):
        attribs = {'id': str(id_str), 'parent': str(parent)}
        if vertex: attribs['vertex'] = str(vertex)
        if edge: attribs['edge'] = str(edge)
        if source: attribs['source'] = str(source)
        if target: attribs['target'] = str(target)
        attribs['value'] = val
        attribs['style'] = style
        c = ET.SubElement(root_cell, 'mxCell', **attribs)
        geom = ET.SubElement(c, 'mxGeometry')
        if edge:
            geom.set('relative', '1')
            geom.set('as', 'geometry')
        else:
            geom.set('x', str(x))
            geom.set('y', str(y))
            geom.set('width', str(w))
            geom.set('height', str(h))
            geom.set('as', 'geometry')
        return str(id_str)

    title_style = "text;html=1;align=center;verticalAlign=middle;fontSize=18;fontStyle=1;fontFamily=Helvetica;fontColor=#1A2530;"
    add_cell(2, "ConvTransformer Block Architecture with Local Window Self-Attention", title_style, 250, 15, 900, 30)

    # Styles
    c_block = "swimlane;startSize=28;horizontal=1;container=1;collapsible=0;rounded=1;arcSize=6;strokeWidth=1.5;fillColor=#F8FAFC;strokeColor=#475569;fontFamily=Helvetica;fontSize=12;fontStyle=1;align=center;fontColor=#0F172A;"
    c_wmsa = "swimlane;startSize=24;horizontal=1;container=1;collapsible=0;rounded=1;arcSize=6;strokeWidth=1.2;dashed=1;fillColor=#EFF6FF;strokeColor=#3B82F6;fontFamily=Helvetica;fontSize=11;fontStyle=1;align=center;fontColor=#1E40AF;"
    c_mlp = "swimlane;startSize=24;horizontal=1;container=1;collapsible=0;rounded=1;arcSize=6;strokeWidth=1.2;dashed=1;fillColor=#FFF7ED;strokeColor=#F97316;fontFamily=Helvetica;fontSize=11;fontStyle=1;align=center;fontColor=#9A3412;"

    b_io = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#64748B;strokeWidth=1.5;fontFamily=Helvetica;fontSize=11;fontStyle=1;align=center;fontColor=#1E293B;"
    b_norm = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FEF3C7;strokeColor=#D97706;strokeWidth=1.5;fontFamily=Helvetica;fontSize=10;fontStyle=1;align=center;fontColor=#92400E;"
    b_op = "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#DBEAFE;strokeColor=#2563EB;strokeWidth=1.2;fontFamily=Helvetica;fontSize=10;align=center;fontColor=#1E40AF;"
    b_mlp_op = "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#FFEDD5;strokeColor=#EA580C;strokeWidth=1.2;fontFamily=Helvetica;fontSize=10;align=center;fontColor=#9A3412;"
    b_add = "ellipse;whiteSpace=wrap;html=1;fillColor=#E2E8F0;strokeColor=#475569;strokeWidth=1.5;fontFamily=Helvetica;fontSize=16;fontStyle=1;align=center;fontColor=#0F172A;"

    edge_main = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#1E293B;strokeWidth=2;endArrow=classic;endFill=1;"
    edge_skip = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#2563EB;strokeWidth=2.2;endArrow=classic;endFill=1;"

    # Main block container
    add_cell(10, "ConvTransformerBlock (combi_seg_den.py : Lines 86–108)", c_block, 150, 60, 1100, 580)

    # Input Tensor
    add_cell(11, "&lt;b&gt;Input Feature Map X&lt;/b&gt;&lt;br&gt;Shape: (B, C, H, W)", b_io, 500, 105, 200, 45)

    # --- Part 1: Attention Sub-Block ---
    add_cell(20, "Window-based Multi-Head Self-Attention (W-MSA / SW-MSA)", c_wmsa, 190, 175, 460, 360)
    add_cell(21, "&lt;b&gt;Group Normalization&lt;/b&gt;&lt;br&gt;GroupNorm(1, C)&lt;br&gt;&lt;font color='#6B7280'&gt;Per-channel spatial norm&lt;/font&gt;", b_norm, 230, 215, 180, 50)
    add_cell(22, "&lt;b&gt;Window Partitioning&lt;/b&gt;&lt;br&gt;Window size M × M (e.g. 8×8 or 4×4)&lt;br&gt;Optional cyclic shift (shift=M//2)", b_op, 230, 280, 180, 50)
    add_cell(23, "&lt;b&gt;Linear Projections (Q, K, V)&lt;/b&gt;&lt;br&gt;dim → 3 × dim (h heads, d = C/h)", b_op, 230, 345, 180, 45)
    add_cell(24, "&lt;b&gt;Scaled Dot-Product Attention&lt;/b&gt;&lt;br&gt;Attn = Softmax((Q Kᵀ)/√d + &lt;b&gt;B&lt;/b&gt;) V&lt;br&gt;&lt;font color='#D97706'&gt;&lt;b&gt;B&lt;/b&gt;: Relative Position Bias Table&lt;/font&gt;", b_op, 210, 405, 220, 55)
    add_cell(25, "&lt;b&gt;Window Un-partitioning &amp; Proj&lt;/b&gt;&lt;br&gt;Reverse shift + Linear proj (dim → dim)", b_op, 230, 475, 180, 45)

    # Residual 1
    add_cell(26, "⊕", b_add, 575, 335, 50, 50)

    # --- Part 2: MLP Sub-Block ---
    add_cell(30, "Multi-Layer Perceptron (MLP Feed-Forward Network)", c_mlp, 730, 175, 460, 360)
    add_cell(31, "&lt;b&gt;Group Normalization&lt;/b&gt;&lt;br&gt;GroupNorm(1, C)&lt;br&gt;&lt;font color='#6B7280'&gt;Per-channel spatial norm&lt;/font&gt;", b_norm, 770, 215, 180, 50)
    add_cell(32, "&lt;b&gt;Pointwise Conv 1×1&lt;/b&gt;&lt;br&gt;Channel expansion (C → 4C)&lt;br&gt;Kernel: 1×1, stride=1", b_mlp_op, 770, 280, 180, 50)
    add_cell(33, "&lt;b&gt;GELU Activation&lt;/b&gt;&lt;br&gt;Gaussian Error Linear Unit&lt;br&gt;Smooth non-linearity", b_mlp_op, 770, 345, 180, 45)
    add_cell(34, "&lt;b&gt;Pointwise Conv 1×1&lt;/b&gt;&lt;br&gt;Channel reduction (4C → C)&lt;br&gt;Kernel: 1×1, stride=1", b_mlp_op, 770, 405, 180, 50)
    add_cell(35, "&lt;b&gt;Feed-Forward Feature&lt;/b&gt;&lt;br&gt;Output shape: (B, C, H, W)", b_mlp_op, 770, 475, 180, 45)

    # Residual 2
    add_cell(36, "⊕", b_add, 1115, 335, 50, 50)

    # Output Tensor
    add_cell(40, "&lt;b&gt;Output Feature Map Y&lt;/b&gt;&lt;br&gt;Shape: (B, C, H, W)", b_io, 850, 580, 200, 45)

    # Forward arrows
    add_cell(201, "", edge_main, 0, 0, 0, 0, edge=1, source=11, target=21)
    add_cell(202, "", edge_main, 0, 0, 0, 0, edge=1, source=21, target=22)
    add_cell(203, "", edge_main, 0, 0, 0, 0, edge=1, source=22, target=23)
    add_cell(204, "", edge_main, 0, 0, 0, 0, edge=1, source=23, target=24)
    add_cell(205, "", edge_main, 0, 0, 0, 0, edge=1, source=24, target=25)
    add_cell(206, "", edge_main, 0, 0, 0, 0, edge=1, source=25, target=26)

    # Residual connection 1: X -> +
    add_cell(207, "Residual X", edge_skip, 0, 0, 0, 0, edge=1, source=11, target=26)

    # Intermediate link to MLP
    add_cell(208, "X'", edge_main, 0, 0, 0, 0, edge=1, source=26, target=31)
    add_cell(209, "", edge_main, 0, 0, 0, 0, edge=1, source=31, target=32)
    add_cell(210, "", edge_main, 0, 0, 0, 0, edge=1, source=32, target=33)
    add_cell(211, "", edge_main, 0, 0, 0, 0, edge=1, source=33, target=34)
    add_cell(212, "", edge_main, 0, 0, 0, 0, edge=1, source=34, target=35)
    add_cell(213, "", edge_main, 0, 0, 0, 0, edge=1, source=35, target=36)

    # Residual connection 2: X' -> +
    add_cell(214, "Residual X'", edge_skip, 0, 0, 0, 0, edge=1, source=26, target=36)

    # Final to output
    add_cell(215, "", edge_main, 0, 0, 0, 0, edge=1, source=36, target=40)

    xml_str = minidom.parseString(ET.tostring(root, encoding='utf-8')).toprettyxml(indent="  ")
    return xml_str

# ==============================================================================
# 3. BUILD DRAW.IO XML FOR FIGURE 4: MULTI-SCALE SEGMENTATION FPN
# ==============================================================================
def build_fig4_drawio_xml():
    root = ET.Element('mxfile', host='app.diagrams.net', modified='2026-10-07T08:00:00.000Z', agent='Antigravity', version='24.0.0', type='device')
    diagram = ET.SubElement(root, 'diagram', id='fig4_fpn_segmentation', name='Figure 4 - Multi-Scale FPN Segmentation Pathway')
    model = ET.SubElement(diagram, 'mxGraphModel', dx='1600', dy='950', grid='1', gridSize='10', guides='1', tooltips='1', connect='1', arrows='1', fold='1', page='1', pageScale='1', pageWidth='1600', pageHeight='850', background='#FFFFFF', math='1', shadow='0')
    root_cell = ET.SubElement(model, 'root')
    ET.SubElement(root_cell, 'mxCell', id='0')
    ET.SubElement(root_cell, 'mxCell', id='1', parent='0')

    def add_cell(id_str, val, style, x, y, w, h, parent='1', vertex='1', edge=None, source=None, target=None):
        attribs = {'id': str(id_str), 'parent': str(parent)}
        if vertex: attribs['vertex'] = str(vertex)
        if edge: attribs['edge'] = str(edge)
        if source: attribs['source'] = str(source)
        if target: attribs['target'] = str(target)
        attribs['value'] = val
        attribs['style'] = style
        c = ET.SubElement(root_cell, 'mxCell', **attribs)
        geom = ET.SubElement(c, 'mxGeometry')
        if edge:
            geom.set('relative', '1')
            geom.set('as', 'geometry')
        else:
            geom.set('x', str(x))
            geom.set('y', str(y))
            geom.set('width', str(w))
            geom.set('height', str(h))
            geom.set('as', 'geometry')
        return str(id_str)

    title_style = "text;html=1;align=center;verticalAlign=middle;fontSize=18;fontStyle=1;fontFamily=Helvetica;fontColor=#1A2530;"
    add_cell(2, "Multi-Scale Feature Pyramid Network (FPN) and Segmentation Head Architecture", title_style, 250, 15, 1100, 30)

    # Styles
    c_sec = "swimlane;startSize=26;horizontal=1;container=1;collapsible=0;rounded=1;arcSize=6;strokeWidth=1.5;fillColor=#F8FAFC;strokeColor=#475569;fontFamily=Helvetica;fontSize=11;fontStyle=1;align=center;fontColor=#0F172A;"
    b_feat = "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#E0E7FF;strokeColor=#4338CA;strokeWidth=1.5;fontFamily=Helvetica;fontSize=10;fontStyle=1;align=center;fontColor=#312E81;"
    b_lat = "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#EDE9FE;strokeColor=#7C3AED;strokeWidth=1.5;fontFamily=Helvetica;fontSize=10;align=center;fontColor=#4C1D95;"
    b_td = "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#F3E8FF;strokeColor=#9333EA;strokeWidth=1.5;fontFamily=Helvetica;fontSize=10;fontStyle=1;align=center;fontColor=#581C87;"
    b_gate = "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#FEF3C7;strokeColor=#D97706;strokeWidth=1.2;fontFamily=Helvetica;fontSize=10;align=center;fontColor=#78350F;"
    b_fuse = "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#DCFCE7;strokeColor=#16A34A;strokeWidth=1.5;fontFamily=Helvetica;fontSize=10;align=center;fontColor=#14532D;"
    b_concat = "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#CCFBF1;strokeColor=#0D9488;strokeWidth=1.8;fontFamily=Helvetica;fontSize=11;fontStyle=1;align=center;fontColor=#115E59;"
    b_head = "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#FFEDD5;strokeColor=#EA580C;strokeWidth=1.5;fontFamily=Helvetica;fontSize=10;align=center;fontColor=#7C2D12;"
    b_out = "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#FEE2E2;strokeColor=#DC2626;strokeWidth=1.8;fontFamily=Helvetica;fontSize=11;fontStyle=1;align=center;fontColor=#7F1D1D;"

    edge_main = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#1E293B;strokeWidth=1.8;endArrow=classic;endFill=1;"
    edge_up = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#7C3AED;strokeWidth=2.2;dashed=1;dashPattern=5 3;endArrow=classic;endFill=1;"

    # 1. Hierarchical Feature Taps
    add_cell(10, "1. Hierarchical Encoder Taps", c_sec, 30, 60, 220, 570)
    add_cell(11, "&lt;b&gt;Stem Representation F1&lt;/b&gt;&lt;br&gt;Shape: 32 × H × W&lt;br&gt;&lt;font color='#4338CA'&gt;High-resolution details&lt;/font&gt;", b_feat, 45, 110, 190, 65)
    add_cell(12, "&lt;b&gt;Down1 Representation F2&lt;/b&gt;&lt;br&gt;Shape: 64 × H/2 × W/2&lt;br&gt;&lt;font color='#4338CA'&gt;Intermediate semantics&lt;/font&gt;", b_feat, 45, 270, 190, 65)
    add_cell(13, "&lt;b&gt;Down2 Representation F3&lt;/b&gt;&lt;br&gt;Shape: 128 × H/4 × W/4&lt;br&gt;&lt;font color='#4338CA'&gt;Coarse contextual features&lt;/font&gt;", b_feat, 45, 430, 190, 65)

    # 2. Lateral Projections (mid_ch = 256)
    add_cell(20, "2. Lateral 1×1 Convolutions", c_sec, 275, 60, 210, 570)
    add_cell(21, "&lt;b&gt;Lateral Conv 1&lt;/b&gt;&lt;br&gt;Conv 1×1 (32 → 256)&lt;br&gt;&lt;b&gt;P1&lt;/b&gt; (256 × H × W)", b_lat, 290, 110, 180, 65)
    add_cell(22, "&lt;b&gt;Lateral Conv 2&lt;/b&gt;&lt;br&gt;Conv 1×1 (64 → 256)&lt;br&gt;&lt;b&gt;P2&lt;/b&gt; (256 × H/2 × W/2)", b_lat, 290, 270, 180, 65)
    add_cell(23, "&lt;b&gt;Lateral Conv 3&lt;/b&gt;&lt;br&gt;Conv 1×1 (128 → 256)&lt;br&gt;&lt;b&gt;P3&lt;/b&gt; (256 × H/4 × W/4)", b_lat, 290, 430, 180, 65)

    # 3. Top-Down Aggregation & Channel Gating
    add_cell(30, "3. Top-Down Fusion &amp; Channel Gating", c_sec, 510, 60, 260, 570)
    add_cell(31, "&lt;b&gt;Level 1 Top-Down Sum&lt;/b&gt;&lt;br&gt;P̃1 = P1 + BilinearUp_2×(P̃2)&lt;br&gt;Shape: 256 × H × W", b_td, 525, 110, 230, 60)
    add_cell(32, "&lt;b&gt;Channel Gate 1&lt;/b&gt;&lt;br&gt;GAP → Conv 1×1 → Sigmoid&lt;br&gt;Recalibration: P̃1 ⊙ g1", b_gate, 525, 180, 230, 50)

    add_cell(33, "&lt;b&gt;Level 2 Top-Down Sum&lt;/b&gt;&lt;br&gt;P̃2 = P2 + BilinearUp_2×(P3)&lt;br&gt;Shape: 256 × H/2 × W/2", b_td, 525, 270, 230, 60)
    add_cell(34, "&lt;b&gt;Channel Gate 2&lt;/b&gt;&lt;br&gt;GAP → Conv 1×1 → Sigmoid&lt;br&gt;Recalibration: P̃2 ⊙ g2", b_gate, 525, 340, 230, 50)

    add_cell(35, "&lt;b&gt;Level 3 Feature&lt;/b&gt;&lt;br&gt;P̃3 = P3&lt;br&gt;Shape: 256 × H/4 × W/4", b_td, 525, 430, 230, 60)
    add_cell(36, "&lt;b&gt;Channel Gate 3&lt;/b&gt;&lt;br&gt;GAP → Conv 1×1 → Sigmoid&lt;br&gt;Recalibration: P̃3 ⊙ g3", b_gate, 525, 500, 230, 50)

    # 4. Fusion Convolutions & Spatial Upsampling
    add_cell(40, "4. Fusion Convs (256→128) &amp; Alignment", c_sec, 795, 60, 250, 570)
    add_cell(41, "&lt;b&gt;Fusion 1 (Level 1)&lt;/b&gt;&lt;br&gt;Conv 3×3 (256 → 128) + GN(8) + GELU&lt;br&gt;&lt;b&gt;G1&lt;/b&gt; (128 × H × W)", b_fuse, 810, 130, 220, 65)
    add_cell(42, "&lt;b&gt;Fusion 2 (Level 2)&lt;/b&gt;&lt;br&gt;BilinearUp_2× to H × W&lt;br&gt;Conv 3×3 (256 → 128) + GN(8) + GELU&lt;br&gt;&lt;b&gt;G2&lt;/b&gt; (128 × H × W)", b_fuse, 810, 290, 220, 75)
    add_cell(43, "&lt;b&gt;Fusion 3 (Level 3)&lt;/b&gt;&lt;br&gt;BilinearUp_4× to H × W&lt;br&gt;Conv 3×3 (256 → 128) + GN(8) + GELU&lt;br&gt;&lt;b&gt;G3&lt;/b&gt; (128 × H × W)", b_fuse, 810, 450, 220, 75)

    # 5. Concatenation & Segmentation Head
    add_cell(50, "5. Fused Representation &amp; Segmentation Head", c_sec, 1070, 60, 490, 570)
    add_cell(51, "&lt;b&gt;Multi-Scale Channel Concatenation&lt;/b&gt;&lt;br&gt;torch.cat([G1, G2, G3], dim=1)&lt;br&gt;&lt;b&gt;F_seg&lt;/b&gt; (384 × H × W)", b_concat, 1090, 100, 450, 55)

    add_cell(52, "&lt;b&gt;Layer 0: ConvTransformerBlock&lt;/b&gt;&lt;br&gt;384 channels, 12 heads (head_dim=32), window=8&lt;br&gt;Captures long-range spatial plume dependencies", b_head, 1090, 175, 450, 55)
    add_cell(53, "&lt;b&gt;Layer 1: Grouped Convolution 3×3&lt;/b&gt;&lt;br&gt;in=384, out=384, groups=128 + GELU&lt;br&gt;Efficient local feature refinement", b_head, 1090, 250, 450, 50)
    add_cell(54, "&lt;b&gt;Layer 3: Progressive Channel Reduction 1&lt;/b&gt;&lt;br&gt;Conv 1×1 (384 → 256)&lt;br&gt;Reduces feature dimensionality (-33%)", b_head, 1090, 320, 450, 50)
    add_cell(55, "&lt;b&gt;Layer 4: Progressive Channel Reduction 2&lt;/b&gt;&lt;br&gt;Conv 3×3 (256 → 128, padding=1)&lt;br&gt;Smoothing and intermediate spatial mixing (-50%)", b_head, 1090, 390, 450, 50)
    add_cell(56, "&lt;b&gt;Layer 5 &amp; 6: Final Projection &amp; Activation&lt;/b&gt;&lt;br&gt;Conv 1×1 (128 → 1) + Sigmoid()&lt;br&gt;Per-pixel probability estimation", b_head, 1090, 460, 450, 50)

    add_cell(57, "&lt;b&gt;Predicted Plume Probability Map (M̂)&lt;/b&gt;&lt;br&gt;Shape: (B, 1, H, W) ∈ [0, 1]&lt;br&gt;Thresholding at τ = 0.5 yields binary plume footprint", b_out, 1090, 535, 450, 65)

    # Connections
    add_cell(101, "", edge_main, 0, 0, 0, 0, edge=1, source=11, target=21)
    add_cell(102, "", edge_main, 0, 0, 0, 0, edge=1, source=12, target=22)
    add_cell(103, "", edge_main, 0, 0, 0, 0, edge=1, source=13, target=23)

    add_cell(104, "", edge_main, 0, 0, 0, 0, edge=1, source=21, target=31)
    add_cell(105, "", edge_main, 0, 0, 0, 0, edge=1, source=22, target=33)
    add_cell(106, "", edge_main, 0, 0, 0, 0, edge=1, source=23, target=35)

    # Top-down arrows
    add_cell(107, "Bilinear 2×", edge_up, 0, 0, 0, 0, edge=1, source=35, target=33)
    add_cell(108, "Bilinear 2×", edge_up, 0, 0, 0, 0, edge=1, source=33, target=31)

    # Gating arrows
    add_cell(109, "", edge_main, 0, 0, 0, 0, edge=1, source=31, target=32)
    add_cell(110, "", edge_main, 0, 0, 0, 0, edge=1, source=33, target=34)
    add_cell(111, "", edge_main, 0, 0, 0, 0, edge=1, source=35, target=36)

    # To fusions
    add_cell(112, "", edge_main, 0, 0, 0, 0, edge=1, source=32, target=41)
    add_cell(113, "", edge_main, 0, 0, 0, 0, edge=1, source=34, target=42)
    add_cell(114, "", edge_main, 0, 0, 0, 0, edge=1, source=36, target=43)

    # To concatenation
    add_cell(115, "", edge_main, 0, 0, 0, 0, edge=1, source=41, target=51)
    add_cell(116, "", edge_main, 0, 0, 0, 0, edge=1, source=42, target=51)
    add_cell(117, "", edge_main, 0, 0, 0, 0, edge=1, source=43, target=51)

    # Sequential in head
    add_cell(118, "", edge_main, 0, 0, 0, 0, edge=1, source=51, target=52)
    add_cell(119, "", edge_main, 0, 0, 0, 0, edge=1, source=52, target=53)
    add_cell(120, "", edge_main, 0, 0, 0, 0, edge=1, source=53, target=54)
    add_cell(121, "", edge_main, 0, 0, 0, 0, edge=1, source=54, target=55)
    add_cell(122, "", edge_main, 0, 0, 0, 0, edge=1, source=55, target=56)
    add_cell(123, "", edge_main, 0, 0, 0, 0, edge=1, source=56, target=57)

    xml_str = minidom.parseString(ET.tostring(root, encoding='utf-8')).toprettyxml(indent="  ")
    return xml_str

# ==============================================================================
# 4. RENDER HIGH-RESOLUTION PUBLICATION-GRADE FIGURES (300 DPI)
# ==============================================================================

def render_fig1_png(out_path):
    fig = plt.figure(figsize=(20, 8.5), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42)
    ax.axis('off')

    def draw_box(x, y, w, h, bg, border, lw=1.5, ls='-', radius=0.6, zorder=2):
        box = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0.15,rounding_size={radius}",
            facecolor=bg, edgecolor=border, linewidth=lw, linestyle=ls, zorder=zorder
        )
        ax.add_patch(box)
        return box

    def draw_arrow(x1, y1, x2, y2, color='#2D3748', lw=2.2, ls='-', zorder=5):
        ax.annotate(
            '', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, mutation_scale=14, shrinkA=2, shrinkB=2),
            zorder=zorder
        )

    # Title
    ax.text(50, 40.2, "End-to-End Spaceborne Sensing-to-Inference Workflow for XCO₂ Plume Delineation",
            ha='center', va='center', fontsize=17, fontweight='bold', color='#1A202C')

    # 5 Containers
    stages = [
        {"x": 2, "w": 17.5, "bg": "#F4F7FB", "border": "#2B6CB0", "title": "1. Spaceborne Remote Sensing", "tc": "#1D3557",
         "items": [
             ("Satellite Platform", "CO2M / OCO-2/3 Constellation\nPolar orbit (~705 km alt.)", "#FFFFFF", "#CBD5E0"),
             ("SWIR Spectrometer", "CO₂ Bands (1.6 & 2.06 µm)\n2×2 km² Push-broom swath", "#FFFFFF", "#CBD5E0"),
             ("Atmospheric Retrieval", "Full-physics radiative transfer\nColumn-averaged XCO₂ (ppm)", "#FFFFFF", "#CBD5E0"),
             ("Emission Hotspots", "Point: Boxberg, Lippendorf\nUrban: Berlin, Île-de-France", "#EBF8FF", "#3182CE")
         ]},
        {"x": 21.5, "w": 17.5, "bg": "#FFF8F0", "border": "#C05621", "title": "2. Scene Extraction & Noise", "tc": "#7B341E",
         "items": [
             ("Regional Tile Extraction", "Grid size: H × W (64×64)\nTargeted over source domain", "#FFFFFF", "#CBD5E0"),
             ("Atmospheric Background", "Biogenic spatial variations\nBaseline ~402–415 ppm", "#FFFFFF", "#CBD5E0"),
             ("Observational Noise", "Instrument noise (σ ~ 0.7 ppm)\nResidual cloud/aerosol artifacts", "#FFF5F5", "#E53E3E"),
             ("Noisy Input Tile X", "Single-channel field\nX ∈ ℝ^(B × 1 × H × W)", "#FEEBC8", "#DD6B20")
         ]},
        {"x": 41, "w": 20, "bg": "#F2F9F5", "border": "#276749", "title": "3. Proposed Joint Analytics", "tc": "#1B4332",
         "items": [
             ("Shared ConvTransformer", "Hierarchical tokens (F₁, F₂, F₃)\nLocal-window attention + bias", "#E6FFFA", "#319795"),
             ("Auxiliary Denoising", "Pretrained reconstruction decoder\nPhysical continuity & TV loss", "#FFFFFF", "#CBD5E0"),
             ("Multi-Scale FPN Pathway", "Channel gating (GAP+Sigmoid)\n256→128 Fusion Convs + Head", "#FFFFFF", "#CBD5E0"),
             ("Physics Loss Coupling", "L_total = 0.3 L_den + 0.7 L_seg\nL_phys enforces X_phys ≥ 402 ppm", "#FEFCBF", "#D69E2E")
         ]},
        {"x": 63, "w": 17.5, "bg": "#FAF5FF", "border": "#6B46C1", "title": "4. Calibrated Model Outputs", "tc": "#322659",
         "items": [
             ("Restored Field X̂", "Denoised continuous XCO₂\nPlume enhancements preserved", "#F3E8FF", "#805AD5"),
             ("Probability Map M̂", "Continuous confidence ∈ [0, 1]\nCalibrated boundary uncertainty", "#FAF5FF", "#9F7AEA"),
             ("Binary Footprint", "Thresholded plume mask M_thr\nSharp geometric delineation", "#EDE9FE", "#6B46C1"),
             ("Superior Accuracy", "Lower NWBCE & higher IoU\nacross 5 geographic domains", "#FFFFFF", "#CBD5E0")
         ]},
        {"x": 82.5, "w": 15.5, "bg": "#FEFCBF", "border": "#B7791F", "title": "5. Geospatial Applications", "tc": "#744210",
         "items": [
             ("Flux Inversion", "Mass-balance with wind vectors:\nQ = ∫ U_eff · ΔXCO₂ dl", "#FFFFFF", "#CBD5E0"),
             ("Facility MRV", "Independent validation of\nemission inventories", "#FFFFFF", "#CBD5E0"),
             ("Urban Carbon Budgets", "Separating diffuse city flux\nfrom background biosphere", "#FFFFFF", "#CBD5E0"),
             ("Climate Decision Support", "Inputs to global stocktaking\n& reduction policies", "#FEFCBF", "#B7791F")
         ]}
    ]

    for st in stages:
        draw_box(st["x"], 3.5, st["w"], 34.5, st["bg"], st["border"], lw=1.8, ls='--', radius=0.8, zorder=1)
        ax.text(st["x"] + st["w"]/2, 36.3, st["title"], ha='center', va='center', fontsize=10.5, fontweight='bold', color=st["tc"], zorder=3)
        y_pos = 28.5
        for item in st["items"]:
            draw_box(st["x"] + 1.0, y_pos, st["w"] - 2.0, 6.2, item[2], item[3], lw=1.2, radius=0.4, zorder=2)
            ax.text(st["x"] + st["w"]/2, y_pos + 4.5, item[0], ha='center', va='center', fontsize=8.8, fontweight='bold', color='#1A202C', zorder=3)
            ax.text(st["x"] + st["w"]/2, y_pos + 2.0, item[1], ha='center', va='center', fontsize=7.2, color='#4A5568', zorder=3)
            y_pos -= 7.4

    # Inter-stage arrows
    draw_arrow(19.5, 20.5, 21.5, 20.5, color='#2B6CB0', lw=2.5)
    draw_arrow(39.0, 20.5, 41.0, 20.5, color='#C05621', lw=2.5)
    draw_arrow(61.0, 20.5, 63.0, 20.5, color='#276749', lw=2.5)
    draw_arrow(80.5, 20.5, 82.5, 20.5, color='#6B46C1', lw=2.5)

    # Bottom summary box
    draw_box(2, 0.4, 96, 2.5, '#F8F9FA', '#A0AEC0', lw=1.2, radius=0.3, zorder=2)
    ax.text(50, 1.65, "End-to-End Objective: Recovering subtle plume concentration gradients under observational noise enables robust automated delineation and downstream carbon emission inversion.",
            ha='center', va='center', fontsize=8.5, fontweight='semibold', color='#2D3748', zorder=3)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure 1 successfully rendered to {out_path}")


def render_fig3_png(out_path):
    fig = plt.figure(figsize=(16, 9), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis('off')

    def draw_box(x, y, w, h, bg, border, lw=1.5, ls='-', radius=0.6, zorder=2):
        box = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0.15,rounding_size={radius}",
            facecolor=bg, edgecolor=border, linewidth=lw, linestyle=ls, zorder=zorder
        )
        ax.add_patch(box)
        return box

    def draw_arrow(x1, y1, x2, y2, color='#2D3748', lw=2.0, ls='-', zorder=5):
        ax.annotate(
            '', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, mutation_scale=13, shrinkA=2, shrinkB=2),
            zorder=zorder
        )

    # Title
    ax.text(50, 50.2, "ConvTransformer Block Architecture (Window Attention + Feed-Forward Network)",
            ha='center', va='center', fontsize=16, fontweight='bold', color='#1A202C')

    # Main Container
    draw_box(4, 2.5, 92, 45.5, '#F8FAFC', '#64748B', lw=1.8, radius=1.0, zorder=1)
    ax.text(50, 46.5, "ConvTransformerBlock (dim = C, heads = h, window_size = M)",
            ha='center', va='center', fontsize=12, fontweight='bold', color='#334155', zorder=3)

    # Input Tensor
    draw_box(41, 41, 18, 3.8, '#FFFFFF', '#475569', lw=1.6, radius=0.4, zorder=3)
    ax.text(50, 43.3, "Input Feature Map X", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#1E293B', zorder=4)
    ax.text(50, 42.0, "Shape: (B, C, H, W)", ha='center', va='center', fontsize=8, color='#64748B', zorder=4)

    # Branch 1 Container (Attention)
    draw_box(8, 8, 38, 30.5, '#EFF6FF', '#3B82F6', lw=1.5, ls='--', radius=0.8, zorder=2)
    ax.text(27, 36.8, "Window-based Multi-Head Self-Attention", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#1D4ED8', zorder=3)

    attn_ops = [
        ("Group Normalization", "GroupNorm(1, C)  [LayerNorm over channels]", "#FEF3C7", "#D97706"),
        ("Window Partitioning", "Window size M × M  (optional cyclic shift by M//2)", "#DBEAFE", "#2563EB"),
        ("Linear Projections (Q, K, V)", "dim → 3 × dim (h heads, head_dim d = C/h)", "#DBEAFE", "#2563EB"),
        ("Scaled Dot-Product Attention", "Attn = Softmax((Q Kᵀ)/√d + B) V\nB: Learnable Relative Position Bias Table", "#BFDBFE", "#1D4ED8"),
        ("Window Reverse & Projection", "Reverse cyclic shift + Linear Projection (dim → dim)", "#DBEAFE", "#2563EB")
    ]
    y_pos = 30.5
    for title, desc, bg, bd in attn_ops:
        draw_box(10.5, y_pos, 33, 4.4, bg, bd, lw=1.2, radius=0.4, zorder=3)
        ax.text(27, y_pos + 3.0, title, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1E293B', zorder=4)
        ax.text(27, y_pos + 1.2, desc, ha='center', va='center', fontsize=7.2, color='#374151', zorder=4)
        y_pos -= 5.5

    # Residual 1 Circle
    circle1 = patches.Circle((49, 13), radius=1.8, facecolor='#E2E8F0', edgecolor='#475569', linewidth=1.5, zorder=4)
    ax.add_patch(circle1)
    ax.text(49, 13, "⊕", ha='center', va='center', fontsize=13, fontweight='bold', color='#0F172A', zorder=5)
    ax.text(49, 9.8, "X' = X + Attn(Norm(X))", ha='center', va='center', fontsize=7.8, fontweight='bold', color='#1E40AF', zorder=5)

    # Branch 2 Container (MLP)
    draw_box(54, 8, 38, 30.5, '#FFF7ED', '#F97316', lw=1.5, ls='--', radius=0.8, zorder=2)
    ax.text(73, 36.8, "Multi-Layer Perceptron (Pointwise Feed-Forward)", ha='center', va='center', fontsize=10.5, fontweight='bold', color='#C2410C', zorder=3)

    mlp_ops = [
        ("Group Normalization", "GroupNorm(1, C)  [LayerNorm over channels]", "#FEF3C7", "#D97706"),
        ("Pointwise Convolution 1×1", "Conv2d(C, 4C, kernel=1)  [Channel Expansion]", "#FFEDD5", "#EA580C"),
        ("GELU Non-Linear Activation", "Gaussian Error Linear Unit (Smooth gating)", "#FED7AA", "#DD6B20"),
        ("Pointwise Convolution 1×1", "Conv2d(4C, C, kernel=1)  [Channel Reduction]", "#FFEDD5", "#EA580C"),
        ("Feed-Forward Feature Output", "Shape: (B, C, H, W)", "#FFEDD5", "#EA580C")
    ]
    y_pos = 30.5
    for title, desc, bg, bd in mlp_ops:
        draw_box(56.5, y_pos, 33, 4.4, bg, bd, lw=1.2, radius=0.4, zorder=3)
        ax.text(73, y_pos + 3.0, title, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1E293B', zorder=4)
        ax.text(73, y_pos + 1.2, desc, ha='center', va='center', fontsize=7.2, color='#374151', zorder=4)
        y_pos -= 5.5

    # Residual 2 Circle
    circle2 = patches.Circle((73, 4.5), radius=1.8, facecolor='#E2E8F0', edgecolor='#475569', linewidth=1.5, zorder=4)
    ax.add_patch(circle2)
    ax.text(73, 4.5, "⊕", ha='center', va='center', fontsize=13, fontweight='bold', color='#0F172A', zorder=5)

    # Arrows for Branch 1
    draw_arrow(50, 41, 27, 35, color='#1E293B', lw=1.8)
    draw_arrow(27, 30.5, 27, 29.5, color='#2563EB', lw=1.5)
    draw_arrow(27, 25.0, 27, 24.0, color='#2563EB', lw=1.5)
    draw_arrow(27, 19.5, 27, 18.5, color='#2563EB', lw=1.5)
    draw_arrow(27, 14.0, 27, 13.0, color='#2563EB', lw=1.5)
    draw_arrow(27, 8.5, 47, 13, color='#2563EB', lw=1.8)

    # Residual connection 1 line from input to +
    ax.plot([41, 38, 38, 47], [43, 43, 13, 13], color='#3B82F6', lw=2.0, ls='-', zorder=4)
    ax.text(35, 28, "Residual X", ha='center', va='center', fontsize=8.2, fontweight='bold', color='#2563EB', rotation=90, zorder=5)

    # Intermediate to Branch 2
    draw_arrow(51, 13, 73, 35, color='#1E293B', lw=1.8)
    draw_arrow(73, 30.5, 73, 29.5, color='#EA580C', lw=1.5)
    draw_arrow(73, 25.0, 73, 24.0, color='#EA580C', lw=1.5)
    draw_arrow(73, 19.5, 73, 18.5, color='#EA580C', lw=1.5)
    draw_arrow(73, 14.0, 73, 13.0, color='#EA580C', lw=1.5)
    draw_arrow(73, 8.5, 73, 6.5, color='#EA580C', lw=1.8)

    # Residual connection 2 line from X' to +2
    ax.plot([49, 49, 71], [11.2, 4.5, 4.5], color='#F97316', lw=2.0, ls='-', zorder=4)
    ax.text(58, 5.7, "Residual X'", ha='center', va='center', fontsize=8.2, fontweight='bold', color='#EA580C', zorder=5)

    # Output box
    draw_box(80, 2.7, 14, 3.6, '#FFFFFF', '#15803D', lw=1.6, radius=0.4, zorder=3)
    ax.text(87, 4.9, "Output Y", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#166534', zorder=4)
    ax.text(87, 3.6, "Y ∈ ℝ^(B×C×H×W)", ha='center', va='center', fontsize=7.5, color='#4B5563', zorder=4)
    draw_arrow(75, 4.5, 80, 4.5, color='#16A34A', lw=2.0)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure 3 successfully rendered to {out_path}")


def render_fig4_png(out_path):
    fig = plt.figure(figsize=(22, 11), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 108)
    ax.set_ylim(0, 52)
    ax.axis('off')

    def draw_box(x, y, w, h, bg, border, lw=1.5, ls='-', radius=0.6, zorder=2):
        box = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle=f"round,pad=0.15,rounding_size={radius}",
            facecolor=bg, edgecolor=border, linewidth=lw, linestyle=ls, zorder=zorder
        )
        ax.add_patch(box)
        return box

    def draw_arrow(x1, y1, x2, y2, color='#2D3748', lw=2.0, ls='-', zorder=5):
        ax.annotate(
            '', xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, mutation_scale=13, shrinkA=2, shrinkB=2),
            zorder=zorder
        )

    # Title
    ax.text(54, 50.2, "Multi-Scale Feature Pyramid Network (FPN) and Segmentation Head Pathway",
            ha='center', va='center', fontsize=17, fontweight='bold', color='#1A202C')

    # 5 Column Containers
    cols = [
        {"x": 2, "w": 18, "bg": "#F8FAFC", "bd": "#475569", "title": "1. Shared Encoder Taps"},
        {"x": 22, "w": 17, "bg": "#F5F3FF", "bd": "#7C3AED", "title": "2. Lateral 1×1 Convs"},
        {"x": 41, "w": 21, "bg": "#FAF5FF", "bd": "#9333EA", "title": "3. Top-Down & Channel Gates"},
        {"x": 64, "w": 19, "bg": "#F0FDF4", "bd": "#16A34A", "title": "4. 3×3 Fusion & Alignment"},
        {"x": 85, "w": 21, "bg": "#FFF7ED", "bd": "#EA580C", "title": "5. Segmentation Head"}
    ]

    for c in cols:
        draw_box(c["x"], 2, c["w"], 46, c["bg"], c["bd"], lw=1.8, ls='--', radius=0.8, zorder=1)
        ax.text(c["x"] + c["w"]/2, 46.5, c["title"], ha='center', va='center', fontsize=11, fontweight='bold', color='#1E293B', zorder=3)

    # Row 1: High-res (Stem)
    y1 = 34.5
    draw_box(3.5, y1, 15, 8.5, '#E0E7FF', '#4338CA', lw=1.4, radius=0.4, zorder=2)
    ax.text(11, y1 + 6.3, "Stem Feature F₁", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#312E81', zorder=3)
    ax.text(11, y1 + 4.2, "32 × H × W", ha='center', va='center', fontsize=8.2, fontweight='semibold', color='#3730A3', zorder=3)
    ax.text(11, y1 + 2.0, "High-resolution details\nFine plume edges", ha='center', va='center', fontsize=7.2, color='#4338CA', zorder=3)

    draw_box(23.5, y1 + 1.0, 14, 6.5, '#EDE9FE', '#7C3AED', lw=1.4, radius=0.4, zorder=2)
    ax.text(30.5, y1 + 5.2, "Lateral Conv 1", ha='center', va='center', fontsize=8.8, fontweight='bold', color='#4C1D95', zorder=3)
    ax.text(30.5, y1 + 3.5, "Conv 1×1 (32 → 256)", ha='center', va='center', fontsize=7.6, color='#5B21B6', zorder=3)
    ax.text(30.5, y1 + 2.0, "P₁ (256 × H × W)", ha='center', va='center', fontsize=8.2, fontweight='bold', color='#6D28D9', zorder=3)

    draw_box(42.5, y1 - 0.5, 18, 9.5, '#F3E8FF', '#9333EA', lw=1.4, radius=0.4, zorder=2)
    ax.text(51.5, y1 + 7.2, "Top-Down Sum & Gate", ha='center', va='center', fontsize=8.8, fontweight='bold', color='#581C87', zorder=3)
    ax.text(51.5, y1 + 5.5, "P̃₁ = P₁ + BilinearUp(P̃₂)", ha='center', va='center', fontsize=7.6, color='#6B21A8', zorder=3)
    ax.text(51.5, y1 + 3.8, "Gate: GAP → Conv 1×1 → σ", ha='center', va='center', fontsize=7.4, color='#7E22CE', zorder=3)
    ax.text(51.5, y1 + 1.8, "Recalibrated: P̃₁ ⊙ g₁ (256 ch)", ha='center', va='center', fontsize=7.8, fontweight='bold', color='#9333EA', zorder=3)

    draw_box(65.5, y1 + 0.5, 16, 7.5, '#DCFCE7', '#16A34A', lw=1.4, radius=0.4, zorder=2)
    ax.text(73.5, y1 + 6.0, "Fusion Conv 1", ha='center', va='center', fontsize=8.8, fontweight='bold', color='#14532D', zorder=3)
    ax.text(73.5, y1 + 4.2, "Conv 3×3 (256 → 128)\n+ GroupNorm(8) + GELU", ha='center', va='center', fontsize=7.2, color='#15803D', zorder=3)
    ax.text(73.5, y1 + 1.8, "G₁ (128 × H × W)", ha='center', va='center', fontsize=8.2, fontweight='bold', color='#166534', zorder=3)

    # Row 2: Mid-res (Down1)
    y2 = 19.5
    draw_box(3.5, y2, 15, 8.5, '#E0E7FF', '#4338CA', lw=1.4, radius=0.4, zorder=2)
    ax.text(11, y2 + 6.3, "Down1 Feature F₂", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#312E81', zorder=3)
    ax.text(11, y2 + 4.2, "64 × H/2 × W/2", ha='center', va='center', fontsize=8.2, fontweight='semibold', color='#3730A3', zorder=3)
    ax.text(11, y2 + 2.0, "Intermediate scale\nSpatial context", ha='center', va='center', fontsize=7.2, color='#4338CA', zorder=3)

    draw_box(23.5, y2 + 1.0, 14, 6.5, '#EDE9FE', '#7C3AED', lw=1.4, radius=0.4, zorder=2)
    ax.text(30.5, y2 + 5.2, "Lateral Conv 2", ha='center', va='center', fontsize=8.8, fontweight='bold', color='#4C1D95', zorder=3)
    ax.text(30.5, y2 + 3.5, "Conv 1×1 (64 → 256)", ha='center', va='center', fontsize=7.6, color='#5B21B6', zorder=3)
    ax.text(30.5, y2 + 2.0, "P₂ (256 × H/2 × W/2)", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#6D28D9', zorder=3)

    draw_box(42.5, y2 - 0.5, 18, 9.5, '#F3E8FF', '#9333EA', lw=1.4, radius=0.4, zorder=2)
    ax.text(51.5, y2 + 7.2, "Top-Down Sum & Gate", ha='center', va='center', fontsize=8.8, fontweight='bold', color='#581C87', zorder=3)
    ax.text(51.5, y2 + 5.5, "P̃₂ = P₂ + BilinearUp(P₃)", ha='center', va='center', fontsize=7.6, color='#6B21A8', zorder=3)
    ax.text(51.5, y2 + 3.8, "Gate: GAP → Conv 1×1 → σ", ha='center', va='center', fontsize=7.4, color='#7E22CE', zorder=3)
    ax.text(51.5, y2 + 1.8, "Recalibrated: P̃₂ ⊙ g₂ (256 ch)", ha='center', va='center', fontsize=7.8, fontweight='bold', color='#9333EA', zorder=3)

    draw_box(65.5, y2 - 0.5, 16, 9.5, '#DCFCE7', '#16A34A', lw=1.4, radius=0.4, zorder=2)
    ax.text(73.5, y2 + 7.5, "Fusion Conv 2", ha='center', va='center', fontsize=8.8, fontweight='bold', color='#14532D', zorder=3)
    ax.text(73.5, y2 + 5.8, "Bilinear Upsample 2×", ha='center', va='center', fontsize=7.6, fontweight='bold', color='#2E7D32', zorder=3)
    ax.text(73.5, y2 + 4.0, "Conv 3×3 (256 → 128)\n+ GroupNorm(8) + GELU", ha='center', va='center', fontsize=7.2, color='#15803D', zorder=3)
    ax.text(73.5, y2 + 1.6, "G₂ (128 × H × W)", ha='center', va='center', fontsize=8.2, fontweight='bold', color='#166534', zorder=3)

    # Row 3: Coarse (Down2)
    y3 = 4.5
    draw_box(3.5, y3, 15, 8.5, '#E0E7FF', '#4338CA', lw=1.4, radius=0.4, zorder=2)
    ax.text(11, y3 + 6.3, "Down2 Feature F₃", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#312E81', zorder=3)
    ax.text(11, y3 + 4.2, "128 × H/4 × W/4", ha='center', va='center', fontsize=8.2, fontweight='semibold', color='#3730A3', zorder=3)
    ax.text(11, y3 + 2.0, "Deep semantic context\nGlobal plume presence", ha='center', va='center', fontsize=7.2, color='#4338CA', zorder=3)

    draw_box(23.5, y3 + 1.0, 14, 6.5, '#EDE9FE', '#7C3AED', lw=1.4, radius=0.4, zorder=2)
    ax.text(30.5, y3 + 5.2, "Lateral Conv 3", ha='center', va='center', fontsize=8.8, fontweight='bold', color='#4C1D95', zorder=3)
    ax.text(30.5, y3 + 3.5, "Conv 1×1 (128 → 256)", ha='center', va='center', fontsize=7.6, color='#5B21B6', zorder=3)
    ax.text(30.5, y3 + 2.0, "P₃ (256 × H/4 × W/4)", ha='center', va='center', fontsize=8.0, fontweight='bold', color='#6D28D9', zorder=3)

    draw_box(42.5, y3 - 0.5, 18, 9.5, '#F3E8FF', '#9333EA', lw=1.4, radius=0.4, zorder=2)
    ax.text(51.5, y3 + 7.2, "Bottom Level Gate", ha='center', va='center', fontsize=8.8, fontweight='bold', color='#581C87', zorder=3)
    ax.text(51.5, y3 + 5.5, "P̃₃ = P₃ (No Top-Down)", ha='center', va='center', fontsize=7.6, color='#6B21A8', zorder=3)
    ax.text(51.5, y3 + 3.8, "Gate: GAP → Conv 1×1 → σ", ha='center', va='center', fontsize=7.4, color='#7E22CE', zorder=3)
    ax.text(51.5, y3 + 1.8, "Recalibrated: P̃₃ ⊙ g₃ (256 ch)", ha='center', va='center', fontsize=7.8, fontweight='bold', color='#9333EA', zorder=3)

    draw_box(65.5, y3 - 0.5, 16, 9.5, '#DCFCE7', '#16A34A', lw=1.4, radius=0.4, zorder=2)
    ax.text(73.5, y3 + 7.5, "Fusion Conv 3", ha='center', va='center', fontsize=8.8, fontweight='bold', color='#14532D', zorder=3)
    ax.text(73.5, y3 + 5.8, "Bilinear Upsample 4×", ha='center', va='center', fontsize=7.6, fontweight='bold', color='#2E7D32', zorder=3)
    ax.text(73.5, y3 + 4.0, "Conv 3×3 (256 → 128)\n+ GroupNorm(8) + GELU", ha='center', va='center', fontsize=7.2, color='#15803D', zorder=3)
    ax.text(73.5, y3 + 1.6, "G₃ (128 × H × W)", ha='center', va='center', fontsize=8.2, fontweight='bold', color='#166534', zorder=3)

    # Column 5: Segmentation Head Blocks
    draw_box(86.5, 39, 18, 5.5, '#CCFBF1', '#0D9488', lw=1.6, radius=0.4, zorder=2)
    ax.text(95.5, 42.8, "Multi-Scale Concat", ha='center', va='center', fontsize=9.2, fontweight='bold', color='#115E59', zorder=3)
    ax.text(95.5, 40.8, "torch.cat([G₁, G₂, G₃], dim=1)\nF_seg (384 × H × W)", ha='center', va='center', fontsize=7.6, color='#134E4A', zorder=3)

    head_layers = [
        ("ConvTransformer Block", "384 ch, 12 heads, window 8×8\nCaptures long-range spatial context", "#FFEDD5", "#EA580C"),
        ("Grouped Conv 3×3", "384 ch, groups=128 + GELU\nEfficient spatial refinement", "#FFEDD5", "#EA580C"),
        ("Channel Reduction 1", "Conv 1×1 (384 → 256) [-33%]", "#FED7AA", "#C2410C"),
        ("Intermediate Conv 3×3", "Conv 3×3 (256 → 128, p=1) [-50%]", "#FED7AA", "#C2410C"),
        ("Final Classification", "Conv 1×1 (128 → 1) + Sigmoid()", "#FDBA74", "#9A3412"),
    ]
    y_h = 32.5
    for title, desc, bg, bd in head_layers:
        draw_box(86.5, y_h, 18, 5.0, bg, bd, lw=1.2, radius=0.4, zorder=2)
        ax.text(95.5, y_h + 3.4, title, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#1E293B', zorder=3)
        ax.text(95.5, y_h + 1.5, desc, ha='center', va='center', fontsize=7.0, color='#374151', zorder=3)
        y_h -= 6.0

    # Output Plume Map
    draw_box(86.5, 3.0, 18, 5.5, '#FEE2E2', '#DC2626', lw=1.8, radius=0.4, zorder=2)
    ax.text(95.5, 6.7, "Plume Probability Map M̂", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#991B1B', zorder=3)
    ax.text(95.5, 4.6, "M̂ ∈ [0, 1]^(B × 1 × H × W)\nτ = 0.5 → Binary Footprint", ha='center', va='center', fontsize=7.5, color='#7F1D1D', zorder=3)

    # Lateral Arrows
    draw_arrow(18.5, y1 + 4.2, 23.5, y1 + 4.2, color='#4338CA', lw=1.8)
    draw_arrow(18.5, y2 + 4.2, 23.5, y2 + 4.2, color='#4338CA', lw=1.8)
    draw_arrow(18.5, y3 + 4.2, 23.5, y3 + 4.2, color='#4338CA', lw=1.8)

    draw_arrow(37.5, y1 + 4.2, 42.5, y1 + 4.2, color='#7C3AED', lw=1.8)
    draw_arrow(37.5, y2 + 4.2, 42.5, y2 + 4.2, color='#7C3AED', lw=1.8)
    draw_arrow(37.5, y3 + 4.2, 42.5, y3 + 4.2, color='#7C3AED', lw=1.8)

    # Top-down arrows
    draw_arrow(51.5, y3 + 9.0, 51.5, y2 - 0.5, color='#9333EA', lw=2.2, ls='--')
    ax.text(53.5, (y3 + y2)/2 + 4, "↑ 2× Bilinear", ha='left', va='center', fontsize=7.5, fontweight='bold', color='#7E22CE', zorder=5)

    draw_arrow(51.5, y2 + 9.0, 51.5, y1 - 0.5, color='#9333EA', lw=2.2, ls='--')
    ax.text(53.5, (y2 + y1)/2 + 4, "↑ 2× Bilinear", ha='left', va='center', fontsize=7.5, fontweight='bold', color='#7E22CE', zorder=5)

    # To fusions
    draw_arrow(60.5, y1 + 4.2, 65.5, y1 + 4.2, color='#9333EA', lw=1.8)
    draw_arrow(60.5, y2 + 4.2, 65.5, y2 + 4.2, color='#9333EA', lw=1.8)
    draw_arrow(60.5, y3 + 4.2, 65.5, y3 + 4.2, color='#9333EA', lw=1.8)

    # From fusions to concatenation
    draw_arrow(81.5, y1 + 4.2, 86.5, 41.5, color='#16A34A', lw=1.8)
    draw_arrow(81.5, y2 + 4.2, 86.5, 41.5, color='#16A34A', lw=1.8)
    draw_arrow(81.5, y3 + 4.2, 86.5, 41.5, color='#16A34A', lw=1.8)

    # Within head
    draw_arrow(95.5, 39.0, 95.5, 37.5, color='#0D9488', lw=1.8)
    draw_arrow(95.5, 32.5, 95.5, 31.5, color='#EA580C', lw=1.5)
    draw_arrow(95.5, 26.5, 95.5, 25.5, color='#EA580C', lw=1.5)
    draw_arrow(95.5, 20.5, 95.5, 19.5, color='#EA580C', lw=1.5)
    draw_arrow(95.5, 14.5, 95.5, 13.5, color='#EA580C', lw=1.5)
    draw_arrow(95.5, 8.5, 95.5, 8.5, color='#EA580C', lw=1.8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure 4 successfully rendered to {out_path}")


if __name__ == '__main__':
    # Build drawio XML files
    xml1 = build_fig1_drawio_xml()
    xml3 = build_fig3_drawio_xml()
    xml4 = build_fig4_drawio_xml()

    # Save to Paper/
    with open('Paper/fig1_workflow.drawio', 'w', encoding='utf-8') as f:
        f.write(xml1)
    with open('Paper/fig3_convtransformer.drawio', 'w', encoding='utf-8') as f:
        f.write(xml3)
    with open('Paper/fig4_fpn_segmentation.drawio', 'w', encoding='utf-8') as f:
        f.write(xml4)

    # Save copies to Diya project/
    with open('Diya project/fig1_workflow.drawio', 'w', encoding='utf-8') as f:
        f.write(xml1)
    with open('Diya project/fig3_convtransformer.drawio', 'w', encoding='utf-8') as f:
        f.write(xml3)
    with open('Diya project/fig4_fpn_segmentation.drawio', 'w', encoding='utf-8') as f:
        f.write(xml4)

    print("All .drawio files saved successfully!")

    # Render PNGs
    # Figure 1: Intro.png and fig1_workflow.png
    render_fig1_png('Paper/Intro.png')
    render_fig1_png('Paper/fig1_workflow.png')

    # Figure 3: 2.png and fig3_convtransformer.png
    render_fig3_png('Paper/2.png')
    render_fig3_png('Paper/fig3_convtransformer.png')

    # Figure 4: 3.png and fig4_fpn_segmentation.png
    render_fig4_png('Paper/3.png')
    render_fig4_png('Paper/fig4_fpn_segmentation.png')

    print("All figures rendered and updated successfully!")
