from django.db import models

# Create your models here.
class Movie(models.Model):
    title = models.CharField(max_length=300)
    genres = models.CharField(max_length=100)
    year = models.IntegerField()
    rating = models.FloatField()
    description = models.TextField()

class User(models.Model):
    username = models.CharField(max_length=100)
    name = models.CharField(max_length=150)

class CurrentWatch(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    time = models.DateTimeField()