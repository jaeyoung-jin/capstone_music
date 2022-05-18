from pyexpat import model
from django.db import models

# Create your models here.

# This is an auto-generated Django model module.
# You'll have to do the following manually to clean this up:
#   * Rearrange models' order
#   * Make sure each model has one field with primary_key=True
#   * Make sure each ForeignKey and OneToOneField has `on_delete` set to the desired behavior
#   * Remove `managed = False` lines if you wish to allow Django to create, modify, and delete the table
# Feel free to rename the models, but don't rename db_table values or field names.


class Dataset(models.Model):
    music_id = models.AutoField(primary_key=True)
    musicname = models.CharField(max_length=200)
    length = models.IntegerField(blank=True, null=True)
    rms_mean = models.FloatField(blank=True, null=True)
    rms_var = models.FloatField(blank=True, null=True)
    bpm = models.FloatField(blank=True, null=True)
    zero_crossing_rate_mean = models.FloatField(blank=True, null=True)
    zero_crossing_rate_var = models.FloatField(blank=True, null=True)
    harmony_mean = models.FloatField(blank=True, null=True)
    harmony_var = models.FloatField(blank=True, null=True)
    percussive_mean = models.FloatField(blank=True, null=True)
    percussive_var = models.FloatField(blank=True, null=True)
    spectral_centroid_mean = models.FloatField(blank=True, null=True)
    spectral_centroid_var = models.FloatField(blank=True, null=True)
    spectral_bandwidth_mean = models.FloatField(blank=True, null=True)
    spectral_bandwidth_var = models.FloatField(blank=True, null=True)
    spectral_rolloff_mean = models.FloatField(blank=True, null=True)
    spectral_rolloff_var = models.FloatField(blank=True, null=True)
    chroma_frequencies_mean = models.FloatField(blank=True, null=True)
    chroma_frequencies_var = models.FloatField(blank=True, null=True)
    mfcc1_mean = models.FloatField(blank=True, null=True)
    mfcc1_var = models.FloatField(blank=True, null=True)
    mfcc2_mean = models.FloatField(blank=True, null=True)
    mfcc2_var = models.FloatField(blank=True, null=True)
    mfcc3_mean = models.FloatField(blank=True, null=True)
    mfcc3_var = models.FloatField(blank=True, null=True)
    mfcc4_mean = models.FloatField(blank=True, null=True)
    mfcc4_var = models.FloatField(blank=True, null=True)
    mfcc5_mean = models.FloatField(blank=True, null=True)
    mfcc5_var = models.FloatField(blank=True, null=True)
    mfcc6_mean = models.FloatField(blank=True, null=True)
    mfcc6_var = models.FloatField(blank=True, null=True)
    mfcc7_mean = models.FloatField(blank=True, null=True)
    mfcc7_var = models.FloatField(blank=True, null=True)
    mfcc8_mean = models.FloatField(blank=True, null=True)
    mfcc8_var = models.FloatField(blank=True, null=True)
    mfcc9_mean = models.FloatField(blank=True, null=True)
    mfcc9_var = models.FloatField(blank=True, null=True)
    mfcc10_mean = models.FloatField(blank=True, null=True)
    mfcc10_var = models.FloatField(blank=True, null=True)
    mfcc11_mean = models.FloatField(blank=True, null=True)
    mfcc11_var = models.FloatField(blank=True, null=True)
    mfcc12_mean = models.FloatField(blank=True, null=True)
    mfcc12_var = models.FloatField(blank=True, null=True)
    mfcc13_mean = models.FloatField(blank=True, null=True)
    mfcc13_var = models.FloatField(blank=True, null=True)
    mfcc14_mean = models.FloatField(blank=True, null=True)
    mfcc14_var = models.FloatField(blank=True, null=True)
    mfcc15_mean = models.FloatField(blank=True, null=True)
    mfcc15_var = models.FloatField(blank=True, null=True)
    mfcc16_mean = models.FloatField(blank=True, null=True)
    mfcc16_var = models.FloatField(blank=True, null=True)
    mfcc17_mean = models.FloatField(blank=True, null=True)
    mfcc17_var = models.FloatField(blank=True, null=True)
    mfcc18_mean = models.FloatField(blank=True, null=True)
    mfcc18_var = models.FloatField(blank=True, null=True)
    mfcc19_mean = models.FloatField(blank=True, null=True)
    mfcc19_var = models.FloatField(blank=True, null=True)
    mfcc20_mean = models.FloatField(blank=True, null=True)
    mfcc20_var = models.FloatField(blank=True, null=True)
    # audio_file = models.FileField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'dataset'


class Document(models.Model):
    title = models.CharField(max_length = 500, null = True,blank = True)
    uploadedFile = models.FileField(upload_to = "Uploaded Files/")
    dateTimeOfUpload = models.DateTimeField(auto_now = True)



class Music(models.Model):
    music_title = models.CharField(max_length = 300)
    audio_file = models.FileField()




