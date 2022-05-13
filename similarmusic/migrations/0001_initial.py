# Generated by Django 4.0.4 on 2022-05-08 09:06

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Dataset',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('musicname', models.CharField(max_length=200)),
            ],
            options={
                'db_table': 'Dataset_csv',
            },
        ),
        migrations.CreateModel(
            name='Inputmusic',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('input_musicname', models.CharField(max_length=200)),
            ],
        ),
    ]