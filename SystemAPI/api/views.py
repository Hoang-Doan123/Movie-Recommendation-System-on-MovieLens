from django.shortcuts import render
from rest_framework import generics
from rest_framework.response import Response
from serializer import *
from main.model_selection import *

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

    return zip(names, genres)