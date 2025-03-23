from django.urls import path
from . import views

urlpatterns = [
    path('/movies/<int:movie_id>/recommendations',
         views.get_recommendations(),
         name='get_recommendations'),
    path()
]