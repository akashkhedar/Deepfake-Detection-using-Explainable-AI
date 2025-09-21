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
          Upload an image to analyze whether it's REAL or FAKE
        </Typography>
      </Box>

      {/* Main Content */}
      <Box
        sx={{
          display: "flex",
          flexDirection: "column",
          alignItems: "stretch",
          gap: 4,
        }}
      >
        <Box
          sx={{
            display: "flex",
            flexDirection: isMobile ? "column" : "row",
            justifyContent: "space-evenly",
            alignItems: "flex-start",
            gap: isMobile ? 3 : 4,
          }}
        >
          {/* Left Box - Input Image */}
          <Card
            sx={{
              bgcolor: "background.paper",
              display: "flex",
              flexDirection: "column",
              width: isMobile ? "100%" : "auto",
              maxWidth: isMobile ? "400px" : "none",
            }}
          >
            <CardContent
              sx={{
                p: 3,
                flex: 1,
                display: "flex",
                flexDirection: "column",
                overflow: "hidden", // Prevent overflow
              }}
            >
              <Typography
                variant="h5"
                gutterBottom
                sx={{ mb: 3, flexShrink: 0 }}
              >
                <CloudUpload sx={{ mr: 1, verticalAlign: "middle" }} />
                Upload Image
              </Typography>

              <Box
                sx={{
                  flex: 1,
                  display: "flex",
                  flexDirection: "column",
                  minHeight: 0,
                }}
              >
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
              </Box>

              {!selectedImage && (
                <CardActions
                  sx={{ justifyContent: "center", pt: 3, flexShrink: 0 }}
                >
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

          {/* Right Box - Heatmap */}
          <Card
            sx={{
              bgcolor: "background.paper",
              display: "flex",
              flexDirection: "column",
              width: isMobile ? "100%" : "auto",
              maxWidth: isMobile ? "400px" : "none",
            }}
          >
            <CardContent
              sx={{
                p: 3,
                flex: 1,
                display: "flex",
                flexDirection: "column",
                overflow: "hidden", // Prevent overflow
              }}
            >
              <Typography
                variant="h5"
                gutterBottom
                sx={{ mb: 3, flexShrink: 0 }}
              >
                <Analytics sx={{ mr: 1, verticalAlign: "middle" }} />
                Analysis Results
              </Typography>

              <Box
                sx={{
                  flex: 1,
                  display: "flex",
                  flexDirection: "column",
                }}
              >
                {!selectedImage ? (
                  <WaitingState />
                ) : (
                  <HeatmapDisplay
                    heatmapUrl={heatmapUrl}
                    isAnalyzing={isAnalyzing}
                    prediction={prediction}
                  />
                )}
              </Box>
            </CardContent>
          </Card>
        </Box>

        {/* Bottom Box - Explanation (full width) */}
        <Box>
          {selectedImage && prediction === "error" && (
            <ExplanationBox
              explanation={explanation}
              isAnalyzing={isAnalyzing}
              prediction={prediction}
            />
          )}

          {selectedImage &&
            (prediction === "fake" || prediction === "real") &&
            explanation && (
              <ExplanationBox
                explanation={explanation}
                isAnalyzing={isAnalyzing}
                prediction={prediction}
              />
            )}
        </Box>
      </Box>
    </Container>
  );
};

export default ModelSection;
