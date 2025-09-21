import React from "react";
import {
  Box,
  Typography,
  Button,
  Container,
  Grid,
  Card,
  CardContent,
  CardActions,
  useTheme,
  useMediaQuery,
} from "@mui/material";
import { CloudUpload, Analytics, ImageSearch } from "@mui/icons-material";

// Import modular components
import ImageUploadArea from "./ImageUploadArea";
import ImageDisplay from "./ImageDisplay";
import HeatmapDisplay from "./HeatmapDisplay";
import ExplanationBox from "./ExplanationBox";
import RealImageMessage from "./RealImageMessage";
import WaitingState from "./WaitingState";

// Import custom hook
import useImageAnalysis from "./hooks/useImageAnalysis";

const ModelSection = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("md"));

  // Use custom hook for all analysis logic
  const {
    selectedImage,
    prediction,
    heatmapUrl,
    explanation,
    isAnalyzing,
    handleImageUpload,
    handleFileSelect,
    handleReset,
  } = useImageAnalysis();

  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      {/* Header */}
      <Box sx={{ textAlign: "center", mb: 6 }}>
        <Typography
          variant={isMobile ? "h4" : "h3"}
          component="h1"
          gutterBottom
          sx={{
            fontWeight: "bold",
            background: "linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
          }}
        >
          AI-Powered DeepFake Detection
        </Typography>
        <Typography variant="h6" color="text.secondary" sx={{ mb: 4 }}>
          Upload an image to analyze whether it's real or AI-generated
        </Typography>
      </Box>

      {/* Main Content Grid - Centered */}
      <Box sx={{ display: "flex", justifyContent: "center", mb: 4 }}>
        <Grid container spacing={4} sx={{ maxWidth: "1000px" }}>
          {/* Left Side - Image Upload/Display */}
          <Grid item xs={12} md={6}>
            <Card sx={{ height: "100%", bgcolor: "background.paper" }}>
              <CardContent sx={{ p: 3 }}>
                <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
                  <CloudUpload sx={{ mr: 1, verticalAlign: "middle" }} />
                  Upload Image
                </Typography>

                {!selectedImage ? (
                  <ImageUploadArea onImageSelect={handleFileSelect} />
                ) : (
                  <ImageDisplay
                    image={selectedImage}
                    prediction={prediction}
                    isAnalyzing={isAnalyzing}
                    onReset={handleReset}
                  />
                )}

                {!selectedImage && (
                  <CardActions sx={{ justifyContent: "center", pt: 3 }}>
                    <Button
                      variant="contained"
                      startIcon={<ImageSearch />}
                      size="large"
                      onClick={handleImageUpload}
                      sx={{
                        background:
                          "linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)",
                        px: 4,
                        "&:hover": {
                          background:
                            "linear-gradient(45deg, #FE6B8B 60%, #FF8E53 100%)",
                        },
                      }}
                    >
                      Choose Image
                    </Button>
                  </CardActions>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Right Side - Analysis Results */}
          <Grid item xs={12} md={6}>
            <Card sx={{ height: "100%", bgcolor: "background.paper" }}>
              <CardContent sx={{ p: 3 }}>
                <Typography variant="h5" gutterBottom sx={{ mb: 3 }}>
                  <Analytics sx={{ mr: 1, verticalAlign: "middle" }} />
                  Analysis Results
                </Typography>

                {!selectedImage ? (
                  <WaitingState />
                ) : (
                  <Box>
                    <Typography variant="h6" gutterBottom>
                      Attention Heatmap
                    </Typography>
                    <HeatmapDisplay
                      heatmapUrl={heatmapUrl}
                      isAnalyzing={isAnalyzing}
                    />
                  </Box>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Box>

      {/* Error Section - Full Width Below */}
      {selectedImage && prediction === "error" && (
        <Box sx={{ display: "flex", justifyContent: "center" }}>
          <Box sx={{ width: "100%", maxWidth: "1000px" }}>
            <ExplanationBox
              explanation={explanation}
              isAnalyzing={isAnalyzing}
              prediction={prediction}
            />
          </Box>
        </Box>
      )}

      {/* Explanation Section - Full Width Below */}
      {selectedImage && prediction === "fake" && (
        <Box sx={{ display: "flex", justifyContent: "center" }}>
          <Box sx={{ width: "100%", maxWidth: "1000px" }}>
            <ExplanationBox
              explanation={explanation}
              isAnalyzing={isAnalyzing}
              prediction={prediction}
            />
          </Box>
        </Box>
      )}

      {/* Real Image Message - Full Width Below */}
      {selectedImage && prediction === "real" && (
        <Box sx={{ display: "flex", justifyContent: "center" }}>
          <Box sx={{ width: "100%", maxWidth: "1000px" }}>
            <RealImageMessage />
          </Box>
        </Box>
      )}
    </Container>
  );
};

export default ModelSection;
