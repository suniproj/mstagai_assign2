import pandas as pd
from langchain_core.documents import Document
from .parsing import as_list,nutrient,missing

REQUIRED={"recipe_name","source","url","servings","calories","diet_labels","health_labels","cautions","cuisine_type","meal_type","dish_type","ingredient_lines","ingredients","total_nutrients"}

def num(v):
    try: return float(v)
    except (TypeError,ValueError): return None

def make_documents(df,limit=None):
    absent=REQUIRED-set(df.columns)
    if absent: raise ValueError("Missing CSV columns: "+", ".join(sorted(absent)))
    if limit: df=df.head(limit)
    docs=[]
    for i,row in df.iterrows():
        servings=num(row.get("servings")) or 1.0
        cal=num(row.get("calories"))
        protein,pu=nutrient(row.get("total_nutrients"),"PROCNT")
        fat,fu=nutrient(row.get("total_nutrients"),"FAT")
        carbs,cu=nutrient(row.get("total_nutrients"),"CHOCDF")
        ingredients=as_list(row.get("ingredient_lines")) or as_list(row.get("ingredients"))
        diet=as_list(row.get("diet_labels")); health=as_list(row.get("health_labels"))
        cautions=as_list(row.get("cautions")); meals=as_list(row.get("meal_type"))
        cuisines=as_list(row.get("cuisine_type")); dishes=as_list(row.get("dish_type"))
        name=str(row.get("recipe_name",f"Recipe {i}")); source=str(row.get("source","Unknown"))
        url="" if missing(row.get("url")) else str(row.get("url"))
        lines=[f"Recipe: {name}",f"Source: {source}",f"URL: {url or 'unavailable'}",f"Servings: {servings:g}",
               f"Meal type: {', '.join(map(str,meals)) or 'unknown'}",
               f"Cuisine: {', '.join(map(str,cuisines)) or 'unknown'}",
               f"Dish type: {', '.join(map(str,dishes)) or 'unknown'}",
               f"Diet labels: {', '.join(map(str,diet)) or 'none listed'}",
               f"Health labels: {', '.join(map(str,health)) or 'none listed'}",
               f"Cautions: {', '.join(map(str,cautions)) or 'none listed'}","","Ingredients:",
               *[f"- {x}" for x in ingredients],"","Nutrition:"]
        md={"row_id":int(i),"recipe_name":name[:250],"source":source[:250],"url":url[:1000],
            "meal_type":" | ".join(map(str,meals))[:500],"diet_labels":" | ".join(map(str,diet))[:1000],"health_labels":" | ".join(map(str,health))[:2000]}
        if cal is not None:
            cps=cal/servings; lines += [f"- Calories, whole recipe: {cal:.2f} kcal",f"- Calories per serving: {cps:.2f} kcal"]
            md["calories_per_serving"]=round(cps,2)
        if protein is not None:
            pps=protein/servings; lines += [f"- Protein per serving: {pps:.2f} {pu or ''}"]; md["protein_per_serving"]=round(pps,2)
        if fat is not None: lines += [f"- Fat per serving: {fat/servings:.2f} {fu or ''}"]
        if carbs is not None: lines += [f"- Carbohydrates per serving: {carbs/servings:.2f} {cu or ''}"]
        docs.append(Document(page_content="\n".join(lines),metadata=md))
    return docs
