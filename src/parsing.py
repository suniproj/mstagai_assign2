import ast, json, math
def missing(v):
    return v is None or (isinstance(v,float) and math.isnan(v)) or str(v).strip().lower() in {"","nan","none","null"}
def structured(v, default):
    if missing(v): return default
    if isinstance(v,(list,dict)): return v
    t=str(v).strip()
    for parser in (json.loads, ast.literal_eval):
        try: return parser(t)
        except Exception: pass
    if isinstance(default,list):
        sep="|" if "|" in t else ","
        return [x.strip() for x in t.split(sep) if x.strip()]
    return default
def as_list(v):
    x=structured(v,[])
    return x if isinstance(x,list) else list(x.values()) if isinstance(x,dict) else [str(x)]
def nutrient(v,key):
    x=structured(v,{})
    item=x.get(key,{}) if isinstance(x,dict) else {}
    try: return float(item.get("quantity")), item.get("unit")
    except (TypeError,ValueError): return None,item.get("unit")
