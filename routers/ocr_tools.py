import os, io, json, base64
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, ImageOps

from services.template_boxes import (
    DEFAULT_TEMPLATE,
    save_template_override,
    get_templates,
)
from services.template_boxes import ALLOW_OVERRIDES
from services.pdf_utils import pdf_bytes_to_image_first_page

router = APIRouter(prefix="/ocr-tools", tags=["ocr-tools"])

from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()


def _check_auth(credentials: HTTPBasicCredentials = Depends(security)):
    user = os.getenv("OCR_ANNOTATOR_USER")
    pwd = os.getenv("OCR_ANNOTATOR_PASS")
    if not user or not pwd:
        return
    if not (credentials.username == user and credentials.password == pwd):
        raise HTTPException(status_code=401, detail="Unauthorized")


def _require_enabled():
    if os.getenv("ENABLE_OCR_ANNOTATOR", "0") != "1":
        raise HTTPException(404, "Annotator deshabilitado")
    return True


@router.get("/annotator")
async def annotator_ui(_: bool = Depends(_require_enabled), __=Depends(_check_auth)):
    html = """
    <html><head><meta charset="utf-8"><title>OCR Annotator</title>
    <style>
      body{font-family:ui-sans-serif,system-ui;margin:16px}
      #c{border:1px solid #ccc; max-width:95vw}
      .row{margin:8px 0}
      label{margin-right:8px}
      .box{background:#fff;border:1px solid #ddd;padding:8px;margin-top:8px}
      textarea{width:100%;height:120px}
    </style>
    </head><body>
      <h2>Annotator de plantillas (coords normalizadas)</h2>
      <form class="row" method="post" enctype="multipart/form-data" action="/ocr-tools/annotator">
        <label>Tipo documento:
          <select name="tipo_documento">
            <option value="acta">acta</option>
            <option value="curp">curp</option>
            <option value="ine">ine</option>
          </select>
        </label>
        <label>Archivo:
          <input type="file" name="file" accept=".png,.jpg,.jpeg,.pdf" required>
        </label>
        <button type="submit">Cargar</button>
      </form>
      <p>Sube un archivo para comenzar.</p>
    </body></html>
    """
    return HTMLResponse(html)


@router.post("/annotator")
async def annotator_post(
    tipo_documento: str = Form(...),
    file: UploadFile = File(...),
    _: bool = Depends(_require_enabled),
    __=Depends(_check_auth),
):
    data = await file.read()
    ct = file.content_type or "application/octet-stream"

    if "pdf" in ct.lower():
        img = pdf_bytes_to_image_first_page(data)
    else:
        img = Image.open(io.BytesIO(data))
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    tpl = DEFAULT_TEMPLATE.get(tipo_documento)
    boxes = get_templates().get(tpl, {})

    html = f"""
    <html><head><meta charset="utf-8"><title>Annotator</title>
    <style>
      body{{font-family:ui-sans-serif,system-ui;margin:16px}}
      #c{{border:1px solid #ccc; max-width:95vw}}
      .row{{margin:8px 0}}
      .box{{background:#fff;border:1px solid #ddd;padding:8px;margin-top:8px}}
      textarea{{width:100%;height:160px}}
      #legend span{{display:inline-block;margin-right:8px;padding:2px 6px;border:1px solid #ddd}}
      button{{cursor:pointer}}
    </style>
    </head><body>
      <h3>Annotator — {tipo_documento} — plantilla: {tpl}</h3>

      <div class="row">
        <canvas id="c"></canvas>
      </div>

      <div class="box">
        <b>Coordenadas normalizadas (JSON para template_boxes.py):</b>
        <textarea id="out"></textarea>
        <div class="row">
          <button onclick="copyOut()">Copiar</button>
          <button onclick="saveToServer()">Guardar en servidor</button>
          <span id="msg"></span>
        </div>
      </div>

      <script>
      const img = new Image();
      img.onload = init;
      img.src = "data:image/png;base64,{b64}";

      const templateId = "{tpl}";
      const initial = {json.dumps(boxes)};
      let W=0,H=0, scale=1;
      const rects = []; // {{key, x,y,w,h}} normalizados

      function init(){{
        const c = document.getElementById('c');
        W = img.width; H = img.height;
        const maxW = Math.min(window.innerWidth*0.95, W);
        scale = maxW / W;
        c.width = W*scale; c.height = H*scale;
        const ctx = c.getContext('2d');

        Object.entries(initial).forEach(([k, r]) => {{
          rects.push({{ key:k, x:r[0], y:r[1], w:r[2], h:r[3] }});
        }});
        draw();

        let drag = null;
        c.onmousedown = (e) => {{
          const p = getPos(e);
          const hit = hitTest(p.x, p.y);
          if(hit) drag = {{ idx:hit.idx, mode: e.shiftKey ? 'resize':'move', ox:p.x, oy:p.y }};
        }};
        window.onmouseup = ()=> drag=null;
        window.onmousemove = (e) => {{
          if(!drag) return;
          const p = getPos(e);
          const r = rects[drag.idx];
          if(drag.mode==='move'){{
            const dx = (p.x - drag.ox)/W; const dy = (p.y - drag.oy)/H;
            r.x = clamp(r.x+dx,0,1-r.w); r.y = clamp(r.y+dy,0,1-r.h);
            drag.ox = p.x; drag.oy = p.y;
          }} else {{
            const dx = (p.x - drag.ox)/W; const dy = (p.y - drag.oy)/H;
            r.w = clamp(r.w+dx,0.01,1-r.x); r.h = clamp(r.h+dy,0.01,1-r.y);
            drag.ox = p.x; drag.oy = p.y;
          }}
          draw();
        }};

        function draw(){{ 
        ctx.clearRect(0,0,c.width,c.height); 
        ctx.drawImage(img,0,0,c.width,c.height); 
        ctx.lineWidth = 2; ctx.strokeStyle='#00a'; 
        ctx.font='14px sans-serif'; 
        rects.forEach((r,i)=>{{ 
        const x=r.x*W*scale, y=r.y*H*scale, w=r.w*W*scale, h=r.h*H*scale; 
        ctx.strokeRect(x,y,w,h); 
        ctx.fillStyle='rgba(255,255,255,0.7)'; 
        const tw = ctx.measureText(r.key).width+8; 
        ctx.fillRect(x+2,y+2, tw, 18); 
        ctx.fillStyle='#000'; 
        ctx.fillText(r.key, x+6, y+16); 
        }});
          document.getElementById('out').value = JSON.stringify(
            Object.fromEntries(rects.map(r => [r.key, [round(r.x),round(r.y),round(r.w),round(r.h)]])),
            null, 2
          );
        }}

        function round(v) {{ return Math.round(v*1000)/1000; }}
        function clamp(v,a,b){{ return Math.max(a, Math.min(b, v)); }}
        function getPos(e) {{
          const r = c.getBoundingClientRect();
          return {{ x:(e.clientX-r.left)/scale, y:(e.clientY-r.top)/scale }};
        }}
        function hitTest(px,py){{
          for(let i=rects.length-1;i>=0;--i){{
            const r=rects[i], x=r.x*W, y=r.y*H, w=r.w*W, h=r.h*H;
            if(px>=x && px<=x+w && py>=y && py<=y+h) return {{idx:i}};
          }}
          return null;
        }}

        window.copyOut = function(){{
          navigator.clipboard.writeText(document.getElementById('out').value);
          flash('Copiado ✓');
        }}

        window.saveToServer = async function(){{
          try {{
            const boxes = JSON.parse(document.getElementById('out').value);
            const res = await fetch(`/ocr-tools/templates/${{templateId}}`, {{
              method: 'POST',
              headers: {{ 'Content-Type': 'application/json' }},
              body: JSON.stringify({{ boxes }})
            }});
            if(!res.ok) throw new Error(await res.text());
            flash('Guardado ✓ (recargado)');
          }} catch(e) {{
            flash('Error: '+e.message);
          }}
        }}

        function flash(msg) {{
          const el = document.getElementById('msg');
          el.textContent = ' '+msg;
          setTimeout(()=> el.textContent='', 2000);
        }}
      }}
      </script>
    </body></html>
    """
    return HTMLResponse(html)


@router.get("/templates")
async def list_templates(_: bool = Depends(_require_enabled), __=Depends(_check_auth)):
    return JSONResponse(get_templates())


@router.post("/templates/{template_id}")
async def save_template(
    template_id: str,
    payload: dict,
    _: bool = Depends(_require_enabled),
    __=Depends(_check_auth),
):
    if not ALLOW_OVERRIDES:
        raise HTTPException(403, "Overrides deshabilitados en este entorno")
    boxes = payload.get("boxes")
    if not isinstance(boxes, dict):
        raise HTTPException(400, "Body debe ser { 'boxes': {key:[x,y,w,h], ...} }")
    for k, v in boxes.items():
        if not (isinstance(v, (list, tuple)) and len(v) == 4):
            raise HTTPException(400, f"'{k}' debe ser [x,y,w,h]")
    save_template_override(template_id, boxes)
    return {"status": "ok", "template_id": template_id, "count": len(boxes)}
