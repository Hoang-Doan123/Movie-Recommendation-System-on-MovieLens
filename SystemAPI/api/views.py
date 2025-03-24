from django.shortcuts import render
from django.http import HttpResponse
from rest_framework import generics
from rest_framework.response import Response
from .serializer import *
from main.model_selection import *
import sys
import os

sys.path.append("..")

from all_cells_code import item_based_recommend, recommend_top_k, precision_recall_f1_at_k

# Create your views here.
class MovieList(generics.ListCreateAPIView):
    queryset = Movie.objects.all()
    serializer_class = MovieSerializer

class MovieDetail(generics.RetrieveUpdateDestroyAPIView):
    def get(self, request, *args, **kwargs):
        movie_id = request.query_params.get("movie_id")
        
        if not movie_id:
            movies = Movie.objects.all()
        else:
            movies = Movie.objects.filter(id=movie_id)

        serializer = MovieSerializer(movies, many=True)
        return Response(serializer.data)

def get_recommendations(request, movie_id):
    recommendations = knn_recommendations(movie_id)

    names = [recommendations[i][0] for i in range(len(recommendations))]
    genres = [recommendations[i][1] for i in range(len(recommendations))]

    return HttpResponse(list(zip(names, genres)))

def get_item_recommendations(request, movie_id):
    recommendations = item_based_recommend(movie_id)
    return HttpResponse(recommendations.values)

def get_top_recommendations(request, movie_id):
    recommendations = recommend_top_k(movie_id)
    return HttpResponse(",".join(list(map(str, recommendations))))

def evaluations(request):
    precision, recall, f1 = precision_recall_f1_at_k(5)
    return HttpResponse(f"{precision}, {recall}, {f1}")