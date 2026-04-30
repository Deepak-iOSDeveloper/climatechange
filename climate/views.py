from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .analysis import run_charts, run_prediction
import json, os

MEDIA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "media", "charts")

def index(request):
    return render(request, "climate/index.html", {})

def run_analysis(request):
    try:
        stats = run_charts()
        return JsonResponse({"status": "ok", "stats": stats})
    except Exception as e:
        import traceback; traceback.print_exc()
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

@csrf_exempt
def predict(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST only"}, status=405)
    try:
        payload = json.loads(request.body)
        result  = run_prediction(payload)
        return JsonResponse({"status": "ok", "result": result})
    except Exception as e:
        import traceback; traceback.print_exc()
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

def charts_ready(request):
    names = ["scatter", "line", "bar", "heatmap", "boxplot", "hypothesis", "residuals"]
    return JsonResponse({c: os.path.exists(os.path.join(MEDIA_DIR, f"{c}.png")) for c in names})
