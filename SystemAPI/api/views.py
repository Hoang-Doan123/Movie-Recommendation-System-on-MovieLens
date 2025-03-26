from django.shortcuts import render
from django.http import HttpResponse
from rest_framework import generics
from rest_framework.response import Response
from .serializer import *
from main.model_selection import *
import sys
import os
import nbimporter

sys.path.append("..")

from ipynb.fs.full.Svd_item_cf import recommend_similar_movies

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