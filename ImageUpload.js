import React, { useState } from 'react';
import { Button, CircularProgress, Container, Grid, Typography } from '@material-ui/core';
import { makeStyles } from '@material-ui/core/styles';
import axios from 'axios';

const useStyles = makeStyles((theme) => ({
  container: {
    marginTop: theme.spacing(4),
    marginBottom: theme.spacing(4),
  },
  previewImage: {
    width: '100%',
    height: 'auto',
    marginBottom: theme.spacing(2),
    borderRadius: theme.spacing(1),
    boxShadow: theme.shadows[1],
  },
  uploadButton: {
    marginTop: theme.spacing(2),
  },
  progress: {
    marginTop: theme.spacing(2),
    marginBottom: theme.spacing(2),
  },
  errorMessage: {
    marginTop: theme.spacing(2),
    color: theme.palette.error.main,
  },
  recommendationsContainer: {
    marginTop: theme.spacing(4),
  },
  recommendedImage: {
    width: '100%',
    height: 'auto',
    marginBottom: theme.spacing(2),
    borderRadius: theme.spacing(1),
    boxShadow: theme.shadows[1],
  },
}));

const ImageUpload = () => {
  const classes = useStyles();

  const [selectedImage, setSelectedImage] = useState(null);
  const [previewImage, setPreviewImage] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [error, setError] = useState(null);

  // Function to handle image selection from the input field
  const handleImageChange = (event) => {
    const file = event.target.files[0];
    setSelectedImage(file);
    setPreviewImage(URL.createObjectURL(file));
  };

  // Function to handle the image upload and recommendation process
  const handleUpload = () => {
    if (!selectedImage) {
      return;
    }

    const formData = new FormData();
    formData.append('image', selectedImage);

    setUploading(true);
    setUploadProgress(0);
    setError(null);

    axios
      .post('http://localhost:5000/recommend', formData, {
        headers: {
          'Content-Type': 'multipart/form-data'
        },
        onUploadProgress: (progressEvent) => {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          setUploadProgress(progress);
        }
      })
      .then((response) => {
        console.log(response.data.recommended_images);
        setRecommendations(response.data.recommended_images);
        setSelectedImage(null);
      })
      .catch((error) => {
        console.error(error);
        setError('Error occurred during upload. Please try again.');
      })
      .finally(() => {
        setUploading(false);
      });
  };

  // Function to render recommended images section
  const renderRecommendations = () => {
    if (recommendations.length === 0 && !uploading && !error) {
      return (
        <Grid item xs={12}>
          <Typography variant="body1" align="center">
            No recommendations available
          </Typography>
        </Grid>
      );
    }

    if (recommendations.length > 0) {
      return (
        <Grid item xs={12} className={classes.recommendationsContainer}>
          <Typography variant="h6" gutterBottom>
            Recommended Images
          </Typography>

          <Grid container spacing={2}>
            {recommendations.map((image, index) => (
              <Grid item xs={12} sm={6} md={4} key={index}>
                <img
                  src={`http://localhost:5000/${image}`}
                  alt={`Recommended ${index + 1}`}
                  className={classes.recommendedImage}
                />
              </Grid>
            ))}
          </Grid>
        </Grid>
      );
    }

    return null;
  };

  return (
    <Container maxWidth="sm" className={classes.container}>
      <Typography variant="h5" align="center" gutterBottom>
        Image Upload
      </Typography>

      <Grid container spacing={2} justify="center" alignItems="center">
        <Grid item xs={12}>
          <input type="file" accept="image/*" onChange={handleImageChange} />
        </Grid>

        {previewImage && (
          <Grid item xs={12}>
            <img src={previewImage} alt="Preview" className={classes.previewImage} />
          </Grid>
        )}

        <Grid item xs={12}>
          <Button
            variant="contained"
            color="primary"
            disabled={!selectedImage || uploading}
            onClick={handleUpload}
            className={classes.uploadButton}
          >
            {uploading ? (
              <CircularProgress size={24} />
            ) : (
              'Upload'
            )}
          </Button>
        </Grid>

        {uploading && (
          <Grid item xs={12}>
            <CircularProgress variant="determinate" value={uploadProgress} className={classes.progress} />
          </Grid>
        )}

        {error && (
          <Grid item xs={12}>
            <Typography variant="body1" align="center" className={classes.errorMessage}>
              {error}
            </Typography>
          </Grid>
        )}

        {renderRecommendations()}
      </Grid>
    </Container>
  );
};

export default ImageUpload;
