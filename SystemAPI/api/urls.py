from django.urls import path
from . import views

urlpatterns = [
    path('movies/<int:movie_id>/recommendations',
         views.get_recommendations,
         name='get_recommendations'),
    path("movies/<int:movie_id>/item_recommendations",
         views.get_item_recommendations,
         name="get_item_recommendations"),
    path("movies/<int:movie_id>/top_recommendations",
         views.get_top_recommendations,
         name="get_top_recommendations"),
]