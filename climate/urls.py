from django.urls import path
from . import views
urlpatterns = [
    path('', views.index, name='index'),
    path('api/run/', views.run_analysis, name='run'),
    path('api/predict/', views.predict, name='predict'),
    path('api/ready/', views.charts_ready, name='ready'),
]
