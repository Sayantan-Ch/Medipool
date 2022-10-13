from pyexpat import model
from django.shortcuts import render
from .apps import PredictionConfig
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
import pickle
import numpy as np

# Create your views here.
with open('saved_dictionary.pkl', 'rb') as f:
        data_dict1 = pickle.load(f)


class Disease_Predict(APIView):
    #permission_classes = [IsAuthenticated]
    def post(self, request, format=None):
        data = request.data

        symptoms = data
        symptoms = symptoms.split(",")
        input_data = [0] * len(data_dict1["symptom_index"])
        for symptom in symptoms:
            index = data_dict1["symptom_index"][symptom]
            input_data[index] = 1

        # reshaping the input data and converting it
        # into suitable format for model predictions
        input_data = np.array(input_data).reshape(1,-1)
        model = PredictionConfig.classifier
        # generating individual outputs
        prediction = data_dict1["predictions_classes"][model.predict(input_data)[0]]
        
        # making final prediction by taking mode of all predictions
        predictions = {
            "model": prediction,
        }
        # return predictions

        # keys = []
        # values = []
        # for key in data:
        #     keys.append(key)
        #     values.append(data[key])
        # X = pd.Series(values).to_numpy().reshape(1, -1)
        # loaded_classifier = PredictionConfig.classifier 
        # y_pred = loaded_classifier.predict(X)
        # y_pred = pd.Series(y_pred)
        # target_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        # y_pred = y_pred.map(target_map).to_numpy()
        # response_dict = {"Prediced Iris Species": y_pred[0]}
        return Response(predictions, status=200)

# from rest_framework import status
# from rest_framework.decorators import api_view
# from rest_framework.response import Response

# # Create your views here.
# @api_view(['GET', 'POST'])
# def api_add(request):
#     sum = 0
#     response_dict = {}
#     if request.method == 'GET':
#         # Do nothing
#         pass
#     elif request.method == 'POST':
#         # Add the numbers
#         data = request.data
#         for key in data:
#             sum += data[key]
#         response_dict = {"sum": sum}
#     return Response(response_dict, status=status.HTTP_201_CREATED)