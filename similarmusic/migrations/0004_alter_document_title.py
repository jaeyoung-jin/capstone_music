# Generated by Django 4.0.4 on 2022-05-10 06:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('similarmusic', '0003_document_alter_dataset_table'),
    ]

    operations = [
        migrations.AlterField(
            model_name='document',
            name='title',
            field=models.CharField(max_length=300),
        ),
    ]
