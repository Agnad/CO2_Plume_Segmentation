import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import numpy as np
from PIL import Image

def build_drawio_xml():
    root = ET.Element('mxfile', host='app.diagrams.net', modified='2026-10-07T07:30:00.000Z', agent='Antigravity', version='24.0.0', type='device')
    diagram = ET.SubElement(root, 'diagram', id='joint_denoise_seg_arch', name='Figure 2 - Joint Architecture')
    model = ET.SubElement(diagram, 'mxGraphModel', dx='1800', dy='1000', grid='1', gridSize='10', guides='1', tooltips='1', connect='1', arrows='1', fold='1', page='1', pageScale='1', pageWidth='1920', pageHeight='1080', background='#FFFFFF', math='1', shadow='0')
    root_cell = ET.SubElement(model, 'root')
    ET.SubElement(root_cell, 'mxCell', id='0')
    ET.SubElement(root_cell, 'mxCell', id='1', parent='0')

    cells = []

    def add_cell(id_str, val, style, x, y, w, h, parent='1', vertex='1', edge=None, source=None, target=None):
        attribs = {'id': str(id_str), 'parent': str(parent)}
        if vertex:
            attribs['vertex'] = str(vertex)
        if edge:
            attribs['edge'] = str(edge)
        if source:
            attribs['source'] = str(source)
        if target:
            attribs['target'] = str(target)
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
        cells.append(c)
        return str(id_str)

    # Styles
    title_style = "text;html=1;align=center;verticalAlign=middle;fontSize=20;fontStyle=1;fontFamily=Helvetica;fontColor=#1A2530;"
    container_encoder = "swimlane;startSize=30;horizontal=1;container=1;collapsible=0;rounded=1;arcSize=4;strokeWidth=1.5;dashed=1;dashPattern=6 4;fillColor=#F4F7FC;strokeColor=#4A7BB0;fontFamily=Helvetica;fontSize=13;fontStyle=1;align=center;fontColor=#1D3557;"
    container_decoder = "swimlane;startSize=30;horizontal=1;container=1;collapsible=0;rounded=1;arcSize=4;strokeWidth=1.5;dashed=1;dashPattern=6 4;fillColor=#FFF8F0;strokeColor=#E07A5F;fontFamily=Helvetica;fontSize=13;fontStyle=1;align=center;fontColor=#A83214;"
    container_seg = "swimlane;startSize=30;horizontal=1;container=1;collapsible=0;rounded=1;arcSize=4;strokeWidth=1.5;dashed=1;dashPattern=6 4;fillColor=#F2F9F4;strokeColor=#52B788;fontFamily=Helvetica;fontSize=13;fontStyle=1;align=center;fontColor=#2D6A4F;"
    container_loss = "swimlane;startSize=30;horizontal=1;container=1;collapsible=0;rounded=1;arcSize=4;strokeWidth=1.5;dashed=1;dashPattern=6 4;fillColor=#FDF6F8;strokeColor=#C94C74;fontFamily=Helvetica;fontSize=13;fontStyle=1;align=center;fontColor=#800F2F;"
    
    box_stem = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FCEADE;strokeColor=#F4A261;strokeWidth=1.5;fontFamily=Helvetica;fontSize=11;fontColor=#264653;"
    box_stage = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#E0EEFA;strokeColor=#457B9D;strokeWidth=1.5;fontFamily=Helvetica;fontSize=11;fontColor=#1D3557;"
    box_bottleneck = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#D4E6F1;strokeColor=#2980B9;strokeWidth=2;fontFamily=Helvetica;fontSize=11;fontStyle=1;fontColor=#1B4F72;"
    box_decoder = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#D8F3DC;strokeColor=#52B788;strokeWidth=1.5;fontFamily=Helvetica;fontSize=11;fontColor=#1B4332;"
    box_head = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#EDE0D4;strokeColor=#B08968;strokeWidth=1.5;fontFamily=Helvetica;fontSize=11;fontColor=#3D261A;"
    
    box_fpn = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#EDE7F6;strokeColor=#7E57C2;strokeWidth=1.5;fontFamily=Helvetica;fontSize=11;fontColor=#311B92;"
    box_seg_head = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FFE0B2;strokeColor=#FB8C00;strokeWidth=1.5;fontFamily=Helvetica;fontSize=11;fontColor=#E65100;"
    
    box_loss = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FADBD8;strokeColor=#E74C3C;strokeWidth=1.5;fontFamily=Helvetica;fontSize=11;fontColor=#78281F;"
    box_phys = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#FCF3CF;strokeColor=#F39C12;strokeWidth=2;fontFamily=Helvetica;fontSize=11;fontStyle=1;fontColor=#7D6608;"
    box_joint = "rounded=1;arcSize=8;whiteSpace=wrap;html=1;fillColor=#D5D8DC;strokeColor=#34495E;strokeWidth=2;fontFamily=Helvetica;fontSize=12;fontStyle=1;fontColor=#17202A;"

    badge_frozen = "rounded=1;arcSize=12;whiteSpace=wrap;html=1;fillColor=#EBF5FB;strokeColor=#3498DB;strokeWidth=1;fontFamily=Helvetica;fontSize=10;fontColor=#21618C;"
    badge_train = "rounded=1;arcSize=12;whiteSpace=wrap;html=1;fillColor=#E8F8F5;strokeColor=#1ABC9C;strokeWidth=1;fontFamily=Helvetica;fontSize=10;fontColor=#117A65;"

    img_box = "rounded=1;arcSize=4;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#999999;strokeWidth=1.2;fontFamily=Helvetica;fontSize=11;fontStyle=1;align=center;fontColor=#333333;"

    edge_solid = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#2C3E50;strokeWidth=1.8;endArrow=classic;endFill=1;"
    edge_skip = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#2980B9;strokeWidth=2;endArrow=classic;endFill=1;"
    edge_loss = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#C0392B;strokeWidth=1.8;dashed=1;dashPattern=5 3;endArrow=classic;endFill=1;"
    edge_phys = "edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor=#D35400;strokeWidth=2.2;dashed=1;dashPattern=6 3;endArrow=classic;endFill=1;"

    # Title
    add_cell(2, "Proposed Joint Denoising-Segmentation Framework Architecture", title_style, 380, 20, 1000, 35)

    # 1. Input Image
    add_cell(10, "&lt;b&gt;Noisy XCO&lt;sub&gt;2&lt;/sub&gt; Input X&lt;/b&gt;&lt;br&gt;&lt;font color='#555555'&gt;Single-channel observation&lt;br&gt;(B × 1 × H × W)&lt;/font&gt;", img_box, 30, 120, 150, 140)

    # 2. Shared Hierarchical Encoder Container
    add_cell(20, "Shared Hierarchical Encoder (EfficientDenoiser Backbone)", container_encoder, 210, 75, 760, 275)
    add_cell(21, "Status: Pretrained on Denoising | Main weights frozen (∇=0) | Trainable Norm &amp; Bias (\\theta_{den}^{adapt})", badge_frozen, 230, 105, 500, 22)
    
    # Stem
    add_cell(22, "&lt;b&gt;Stem&lt;/b&gt;&lt;br&gt;Conv 7×7 (s=1, p=3)&lt;br&gt;GroupNorm(8, 32)&lt;br&gt;GELU Activation&lt;br&gt;&lt;b&gt;F&lt;sub&gt;1&lt;/sub&gt;&lt;/b&gt; (32 × H × W)", box_stem, 230, 140, 125, 95)
    # Down 1
    add_cell(23, "&lt;b&gt;Downsample 1&lt;/b&gt;&lt;br&gt;Conv 3×3 (s=2, p=1)&lt;br&gt;2× ConvTransformer&lt;br&gt;(dim=64, heads=4, ws=8)&lt;br&gt;&lt;b&gt;F&lt;sub&gt;2&lt;/sub&gt;&lt;/b&gt; (64 × H/2 × W/2)", box_stage, 385, 140, 150, 95)
    # Down 2
    add_cell(24, "&lt;b&gt;Downsample 2&lt;/b&gt;&lt;br&gt;Conv 3×3 (s=2, p=1)&lt;br&gt;2× ConvTransformer&lt;br&gt;(dim=128, heads=8, ws=4)&lt;br&gt;&lt;b&gt;F&lt;sub&gt;3&lt;/sub&gt;&lt;/b&gt; (128 × H/4 × W/4)", box_stage, 565, 140, 150, 95)
    # Bottleneck
    add_cell(25, "&lt;b&gt;Bottleneck&lt;/b&gt;&lt;br&gt;6× ConvTransformer&lt;br&gt;Shifted-Window Attn&lt;br&gt;(dim=128, heads=8, ws=4)&lt;br&gt;&lt;b&gt;x&lt;sub&gt;b&lt;/sub&gt;&lt;/b&gt; (128 × H/4 × W/4)", box_bottleneck, 745, 140, 150, 95)

    # Connections in Encoder
    add_cell(201, "", edge_solid, 0, 0, 0, 0, edge=1, source=10, target=22)
    add_cell(202, "", edge_solid, 0, 0, 0, 0, edge=1, source=22, target=23)
    add_cell(203, "", edge_solid, 0, 0, 0, 0, edge=1, source=23, target=24)
    add_cell(204, "", edge_solid, 0, 0, 0, 0, edge=1, source=24, target=25)

    # Feature Taps Labels
    add_cell(26, "&lt;b&gt;Feature Tap F&lt;sub&gt;1&lt;/sub&gt;&lt;/b&gt;&lt;br&gt;32 × H × W", badge_frozen, 240, 250, 105, 35)
    add_cell(27, "&lt;b&gt;Feature Tap F&lt;sub&gt;2&lt;/sub&gt;&lt;/b&gt;&lt;br&gt;64 × H/2 × W/2", badge_frozen, 405, 250, 110, 35)
    add_cell(28, "&lt;b&gt;Feature Tap F&lt;sub&gt;3&lt;/sub&gt;&lt;/b&gt;&lt;br&gt;128 × H/4 × W/4", badge_frozen, 585, 250, 110, 35)

    # 3. Denoising Decoder Container
    add_cell(30, "Denoising Decoder (Auxiliary Reconstruction Pathway)", container_decoder, 1000, 75, 590, 275)
    add_cell(31, "Status: Pretrained Backbone | Continuous concentration field regression without activation", badge_frozen, 1020, 105, 480, 22)

    # Up 1
    add_cell(32, "&lt;b&gt;Upsample Stage 1&lt;/b&gt;&lt;br&gt;Nearest Upsample (2×)&lt;br&gt;Conv 3×3 (128→64)&lt;br&gt;2× ConvTransformer&lt;br&gt;&lt;font color='#2980B9'&gt;&lt;b&gt;⊕ Add Skip from F&lt;sub&gt;2&lt;/sub&gt;&lt;/b&gt;&lt;/font&gt;", box_decoder, 1020, 140, 145, 95)
    # Up 2
    add_cell(33, "&lt;b&gt;Upsample Stage 2&lt;/b&gt;&lt;br&gt;Nearest Upsample (2×)&lt;br&gt;Conv 3×3 (64→32)&lt;br&gt;2× ConvTransformer&lt;br&gt;&lt;font color='#2980B9'&gt;&lt;b&gt;⊕ Add Skip from F&lt;sub&gt;1&lt;/sub&gt;&lt;/b&gt;&lt;/font&gt;", box_decoder, 1195, 140, 145, 95)
    # Denoising Head
    add_cell(34, "&lt;b&gt;Denoising Head&lt;/b&gt;&lt;br&gt;Conv 3×3 (32→1, p=1)&lt;br&gt;Linear Output&lt;br&gt;(No Activation)", box_head, 1370, 140, 125, 95)

    # Denoised Output
    add_cell(35, "&lt;b&gt;Denoised Field X̂&lt;/b&gt;&lt;br&gt;&lt;font color='#555555'&gt;Clean XCO&lt;sub&gt;2&lt;/sub&gt; Field&lt;br&gt;(B × 1 × H × W)&lt;/font&gt;", img_box, 1515, 125, 135, 125)

    # Connections in Decoder
    add_cell(301, "", edge_solid, 0, 0, 0, 0, edge=1, source=25, target=32)
    add_cell(302, "", edge_solid, 0, 0, 0, 0, edge=1, source=32, target=33)
    add_cell(303, "", edge_solid, 0, 0, 0, 0, edge=1, source=33, target=34)
    add_cell(304, "", edge_solid, 0, 0, 0, 0, edge=1, source=34, target=35)

    # Skips from F2 and F1 to Decoder
    add_cell(305, "Additive Skip F2", edge_skip, 0, 0, 0, 0, edge=1, source=27, target=32)
    add_cell(306, "Additive Skip F1", edge_skip, 0, 0, 0, 0, edge=1, source=26, target=33)

    # 4. Multi-Scale Segmentation Pathway Container
    add_cell(40, "Multi-Scale Segmentation Pathway (Feature Pyramid &amp; Segmentation Head)", container_seg, 30, 390, 1140, 420)
    add_cell(41, "Status: Fully Trainable Modules (\\theta_{seg}) | Gradient Checkpointing in FP32 | Segmentation Head in FP16", badge_train, 50, 420, 600, 22)

    # FPN Sub-container
    add_cell(42, "Feature Pyramid Network (FPN) with Channel Gating Recalibration", "swimlane;startSize=25;rounded=1;arcSize=6;strokeWidth=1;dashed=1;fillColor=#FFFFFF;strokeColor=#7E57C2;fontFamily=Helvetica;fontSize=11;fontStyle=1;fontColor=#512DA8;", 50, 455, 680, 335)
    
    # Laterals
    add_cell(43, "&lt;b&gt;Lateral 1&lt;/b&gt;&lt;br&gt;Conv 1×1 (32→256)&lt;br&gt;&lt;b&gt;P&lt;sub&gt;1&lt;/sub&gt;&lt;/b&gt; (256 × H × W)", box_fpn, 70, 495, 125, 75)
    add_cell(44, "&lt;b&gt;Lateral 2&lt;/b&gt;&lt;br&gt;Conv 1×1 (64→256)&lt;br&gt;&lt;b&gt;P&lt;sub&gt;2&lt;/sub&gt;&lt;/b&gt; (256 × H/2 × W/2)", box_fpn, 70, 585, 125, 75)
    add_cell(45, "&lt;b&gt;Lateral 3&lt;/b&gt;&lt;br&gt;Conv 1×1 (128→256)&lt;br&gt;&lt;b&gt;P&lt;sub&gt;3&lt;/sub&gt;&lt;/b&gt; (256 × H/4 × W/4)", box_fpn, 70, 675, 125, 75)

    # Top-Down Fusion & Channel Gating
    add_cell(46, "&lt;b&gt;Top-Down Fusion &amp; Channel Gating&lt;/b&gt;&lt;br&gt;P̃&lt;sub&gt;2&lt;/sub&gt; = P&lt;sub&gt;2&lt;/sub&gt; + ↑&lt;sub&gt;2&lt;/sub&gt;(P&lt;sub&gt;3&lt;/sub&gt;)&lt;br&gt;P̃&lt;sub&gt;1&lt;/sub&gt; = P&lt;sub&gt;1&lt;/sub&gt; + ↑&lt;sub&gt;2&lt;/sub&gt;(P̃&lt;sub&gt;2&lt;/sub&gt;)&lt;br&gt;Gate: AdaptiveAvgPool → Conv 1×1 → Sigmoid&lt;br&gt;Recalibration: P̃&lt;sub&gt;i&lt;/sub&gt; ⊙ Gate(P̃&lt;sub&gt;i&lt;/sub&gt;)", box_fpn, 220, 545, 230, 140)

    # Multi-scale Convolutions & Alignment
    add_cell(47, "&lt;b&gt;Fusion Convolutions &amp; Spatial Alignment&lt;/b&gt;&lt;br&gt;Conv 3×3 (256→128) + GN(8) + GELU&lt;br&gt;Yields G&lt;sub&gt;1&lt;/sub&gt;, G&lt;sub&gt;2&lt;/sub&gt;, G&lt;sub&gt;3&lt;/sub&gt; (128 ch each)&lt;br&gt;Bilinear Upsampling: ↑&lt;sub&gt;2&lt;/sub&gt; G&lt;sub&gt;2&lt;/sub&gt;, ↑&lt;sub&gt;4&lt;/sub&gt; G&lt;sub&gt;3&lt;/sub&gt; to H × W&lt;br&gt;&lt;b&gt;Concat [G&lt;sub&gt;1&lt;/sub&gt;, ↑&lt;sub&gt;2&lt;/sub&gt;G&lt;sub&gt;2&lt;/sub&gt;, ↑&lt;sub&gt;4&lt;/sub&gt;G&lt;sub&gt;3&lt;/sub&gt;]&lt;/b&gt;&lt;br&gt;&lt;b&gt;F&lt;sub&gt;seg&lt;/sub&gt;&lt;/b&gt; (384 × H × W)", box_fpn, 480, 545, 230, 140)

    # Connections in FPN
    add_cell(401, "", edge_solid, 0, 0, 0, 0, edge=1, source=26, target=43)
    add_cell(402, "", edge_solid, 0, 0, 0, 0, edge=1, source=27, target=44)
    add_cell(403, "", edge_solid, 0, 0, 0, 0, edge=1, source=28, target=45)
    add_cell(404, "", edge_solid, 0, 0, 0, 0, edge=1, source=43, target=46)
    add_cell(405, "", edge_solid, 0, 0, 0, 0, edge=1, source=44, target=46)
    add_cell(406, "", edge_solid, 0, 0, 0, 0, edge=1, source=45, target=46)
    add_cell(407, "", edge_solid, 0, 0, 0, 0, edge=1, source=46, target=47)

    # Segmentation Head Sub-container
    add_cell(48, "Segmentation Head (ConvTransformer + Grouped Conv + Channel Reduction)", "swimlane;startSize=25;rounded=1;arcSize=6;strokeWidth=1;dashed=1;fillColor=#FFFFFF;strokeColor=#FB8C00;fontFamily=Helvetica;fontSize=11;fontStyle=1;fontColor=#E65100;", 750, 455, 300, 335)
    
    add_cell(49, "&lt;b&gt;ConvTransformer Block&lt;/b&gt;&lt;br&gt;(dim=384, heads=12, win=8)&lt;br&gt;&lt;i&gt;Window Self-Attention&lt;/i&gt;", box_seg_head, 770, 495, 260, 65)
    add_cell(50, "&lt;b&gt;Grouped Convolution&lt;/b&gt;&lt;br&gt;Conv 3×3 (384 ch, groups=128)&lt;br&gt;+ GELU Activation", box_seg_head, 770, 570, 260, 60)
    add_cell(51, "&lt;b&gt;Progressive Channel Reduction&lt;/b&gt;&lt;br&gt;Conv 1×1 (384→256)&#10;Conv 3×3 (256→128, p=1)&#10;Conv 1×1 (128→1)&lt;br&gt;&lt;b&gt;Sigmoid Activation&lt;/b&gt;", box_seg_head, 770, 640, 260, 85)

    add_cell(408, "", edge_solid, 0, 0, 0, 0, edge=1, source=47, target=49)
    add_cell(409, "", edge_solid, 0, 0, 0, 0, edge=1, source=49, target=50)
    add_cell(410, "", edge_solid, 0, 0, 0, 0, edge=1, source=50, target=51)

    # Plume Probability Map Output
    add_cell(52, "&lt;b&gt;Plume Map M̂&lt;/b&gt;&lt;br&gt;&lt;font color='#555555'&gt;Plume Probability Map&lt;br&gt;(B × 1 × H × W)&lt;br&gt;Values ∈ [0, 1]&lt;/font&gt;", img_box, 1070, 570, 140, 130)
    add_cell(411, "", edge_solid, 0, 0, 0, 0, edge=1, source=51, target=52)

    # 5. Physics-Constrained Joint Loss & Optimization Container
    add_cell(60, "Physics-Constrained Joint Loss &amp; Optimization", container_loss, 1200, 390, 520, 420)
    add_cell(61, "Joint Multi-Task Criterion: Total Loss = 0.3 L_den + 0.7 L_seg", badge_train, 1220, 420, 400, 22)

    # Denoising Loss Box
    add_cell(62, "&lt;b&gt;Denoising Objective L&lt;sub&gt;den&lt;/sub&gt;&lt;/b&gt;&lt;br&gt;&lt;b&gt;0.8 MSE + 0.2 (1-SSIM) + 0.05 TV + 0.1 L&lt;sub&gt;mass&lt;/sub&gt;&lt;/b&gt;&lt;br&gt;• Compares Denoised X̂ against Clean Target X&lt;sup&gt;clean&lt;/sup&gt;&lt;br&gt;• Mass conservation over plume region: 1/B ∑ |∑ M⊙X̂ - ∑ M⊙X&lt;sup&gt;clean&lt;/sup&gt;|", box_loss, 1220, 455, 480, 80)

    # Physics Coupling Module Box
    add_cell(63, "&lt;b&gt;Physics Coupling &amp; Concentration Consistency (L&lt;sub&gt;phys&lt;/sub&gt;)&lt;/b&gt;&lt;br&gt;&lt;font color='#935116'&gt;Physical conversion:&lt;/font&gt; X&lt;sub&gt;phys&lt;/sub&gt; = 3.42 X̂ + 402.0 ppm&lt;br&gt;&lt;font color='#935116'&gt;Constraint:&lt;/font&gt; Plume mask M̂ &gt; 0.5 must satisfy X&lt;sub&gt;phys&lt;/sub&gt; ≥ 402.0 ppm&lt;br&gt;&lt;font color='#935116'&gt;Violation Penalty:&lt;/font&gt; ReLU(402.0 - X&lt;sub&gt;phys&lt;/sub&gt; ⊙ M̂) clamped &amp; log-scaled", box_phys, 1220, 545, 480, 85)

    # Segmentation Loss Box
    add_cell(64, "&lt;b&gt;Segmentation Objective L&lt;sub&gt;seg&lt;/sub&gt;&lt;/b&gt;&lt;br&gt;&lt;b&gt;L&lt;sub&gt;FT&lt;/sub&gt; + 0.5 L&lt;sub&gt;phys&lt;/sub&gt; + 0.1 L&lt;sub&gt;TV-mask&lt;/sub&gt;&lt;/b&gt;&lt;br&gt;• Boundary-weighted Focal Tversky Loss (α=0.8, γ=2, band weight=3.0)&lt;br&gt;• Physics constraint L&lt;sub&gt;phys&lt;/sub&gt; couples Denoised X̂ with Plume Mask M̂&lt;br&gt;• Mask total variation regularization L&lt;sub&gt;TV-mask&lt;/sub&gt;", box_loss, 1220, 640, 480, 85)

    # Joint Loss Box
    add_cell(65, "&lt;b&gt;Total Joint Loss &amp; Optimization&lt;/b&gt;&lt;br&gt;&lt;font size='3'&gt;&lt;b&gt;L&lt;sub&gt;total&lt;/sub&gt; = 0.3 L&lt;sub&gt;den&lt;/sub&gt; + 0.7 L&lt;sub&gt;seg&lt;/sub&gt;&lt;/b&gt;&lt;/font&gt;&lt;br&gt;Gradient updates: \\theta_{seg} (Trainable) &amp;amp; \\theta_{den}^{adapt} (Norm/Bias) | \\theta_{den}^{frozen} Frozen (∇=0)", box_joint, 1220, 735, 480, 65)

    # Connections to Loss & Physics
    add_cell(601, "", edge_loss, 0, 0, 0, 0, edge=1, source=35, target=62)
    add_cell(602, "", edge_phys, 0, 0, 0, 0, edge=1, source=35, target=63)
    add_cell(603, "", edge_phys, 0, 0, 0, 0, edge=1, source=52, target=63)
    add_cell(604, "", edge_loss, 0, 0, 0, 0, edge=1, source=52, target=64)
    add_cell(605, "", edge_solid, 0, 0, 0, 0, edge=1, source=63, target=64)
    add_cell(606, "", edge_loss, 0, 0, 0, 0, edge=1, source=62, target=65)
    add_cell(607, "", edge_loss, 0, 0, 0, 0, edge=1, source=64, target=65)

    # Legend
    add_cell(70, "&lt;b&gt;LEGEND &amp; FLOW CONVENTIONS:&lt;/b&gt;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;" + 
             "&lt;font color='#2C3E50'&gt;&lt;b&gt;──►&lt;/b&gt; Forward Feature Flow&lt;/font&gt;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;" +
             "&lt;font color='#2980B9'&gt;&lt;b&gt;──►&lt;/b&gt; Additive Skip Connection&lt;/font&gt;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;" +
             "&lt;font color='#C0392B'&gt;&lt;b&gt;- - ►&lt;/b&gt; Loss &amp; Supervision Flow&lt;/font&gt;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;" +
             "&lt;font color='#D35400'&gt;&lt;b&gt;- - ►&lt;/b&gt; Physics Coupling Flow&lt;/font&gt;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;|&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;" +
             "&lt;span style='background:#EBF5FB;border:1px solid #3498DB;padding:2px 6px;border-radius:4px;'&gt;Pretrained Backbone (Frozen / Adapted)&lt;/span&gt;&amp;nbsp;&amp;nbsp;&amp;nbsp;&amp;nbsp;" +
             "&lt;span style='background:#E8F8F5;border:1px solid #1ABC9C;padding:2px 6px;border-radius:4px;'&gt;Trainable Segmentation Modules&lt;/span&gt;",
             "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#F8F9FA;strokeColor=#BDC3C7;strokeWidth=1;fontFamily=Helvetica;fontSize=10;fontColor=#2C3E50;align=center;", 30, 825, 1690, 35)

    xml_str = minidom.parseString(ET.tostring(root, encoding='utf-8')).toprettyxml(indent="  ")
    return xml_str

if __name__ == '__main__':
    content = build_drawio_xml()
    with open('Paper/architecture.drawio', 'w', encoding='utf-8') as f:
        f.write(content)
    with open('Diya project/architecture.drawio', 'w', encoding='utf-8') as f:
        f.write(content)
    print("architecture.drawio written to Paper and Diya project successfully!")

