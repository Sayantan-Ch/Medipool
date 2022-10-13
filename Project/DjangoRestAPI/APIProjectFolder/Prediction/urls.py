from django.urls import path
import Prediction.views as views

urlpatterns = [
    path('predict/', views.Disease_Predict.as_view(), name = 'predict'),
]