import {
  Box,
  Typography,
  Paper,
  Button,
  Container,
  Grid,
  Card,
  CardContent,
  CardActions,
  Divider,
  Chip,
} from "@mui/material";
import {
  CloudUpload,
  Analytics,
  Psychology,
  ImageSearch,
} from "@mui/icons-material";

const ModelSection = () => {
  return (
    <Container maxWidth="lg" sx={{ py: 4 }}>
      <Box sx={{ textAlign: "center", mb: 4 }}>
        <Typography
          variant="h3"
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
        <Typography variant="h6" color="text.secondary" paragraph>
          Upload an image to analyze whether it's real or AI-generated using our
          ResNet18 model
        </Typography>
      </Box>

      <Grid container spacing={4}>
        {/* Input Section */}
        <Grid item xs={12} md={6}>
          <Card
            sx={{
              height: "100%",
              border: "2px dashed",
              borderColor: "primary.main",
              bgcolor: "background.paper",
              transition: "all 0.3s ease",
              "&:hover": {
                borderColor: "secondary.main",
                transform: "translateY(-2px)",
                boxShadow: 3,
              },
            }}
          >
            <CardContent sx={{ textAlign: "center", p: 4 }}>
              <CloudUpload
                sx={{ fontSize: 60, color: "primary.main", mb: 2 }}
              />
              <Typography variant="h5" gutterBottom>
                Upload Image
              </Typography>
              <Typography variant="body2" color="text.secondary" paragraph>
                Drag and drop an image here or click to select
              </Typography>
              <Box
                sx={{
                  minHeight: 200,
                  bgcolor: "action.hover",
                  borderRadius: 2,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  mb: 2,
                }}
              >
                <Typography variant="body2" color="text.secondary">
                  No image selected
                </Typography>
              </Box>
            </CardContent>
            <CardActions sx={{ justifyContent: "center", pb: 3 }}>
              <Button
                variant="contained"
                startIcon={<ImageSearch />}
                size="large"
                sx={{
                  background:
                    "linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)",
                  "&:hover": {
                    background:
                      "linear-gradient(45deg, #FE6B8B 60%, #FF8E53 100%)",
                  },
                }}
              >
                Choose Image
              </Button>
            </CardActions>
          </Card>
        </Grid>

        {/* Output Section */}
        <Grid item xs={12} md={6}>
          <Card sx={{ height: "100%", bgcolor: "background.paper" }}>
            <CardContent sx={{ p: 4 }}>
              <Box sx={{ display: "flex", alignItems: "center", mb: 3 }}>
                <Analytics
                  sx={{ fontSize: 30, color: "primary.main", mr: 1 }}
                />
                <Typography variant="h5">Analysis Results</Typography>
              </Box>

              <Box sx={{ mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                  Prediction
                </Typography>
                <Chip
                  label="Awaiting Analysis"
                  variant="outlined"
                  sx={{ mb: 2 }}
                />
                <Box
                  sx={{
                    minHeight: 120,
                    bgcolor: "action.hover",
                    borderRadius: 2,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    mb: 2,
                  }}
                >
                  <Typography variant="body2" color="text.secondary">
                    Heatmap will appear here
                  </Typography>
                </Box>
              </Box>

              <Divider sx={{ my: 2 }} />

              <Box>
                <Box sx={{ display: "flex", alignItems: "center", mb: 2 }}>
                  <Psychology
                    sx={{ fontSize: 24, color: "secondary.main", mr: 1 }}
                  />
                  <Typography variant="h6">AI Explanation</Typography>
                </Box>
                <Paper
                  sx={{
                    p: 2,
                    bgcolor: "action.hover",
                    minHeight: 100,
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <Typography
                    variant="body2"
                    color="text.secondary"
                    sx={{ fontStyle: "italic" }}
                  >
                    Upload an image to get detailed analysis and explanation...
                  </Typography>
                </Paper>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Container>
  );
};

export default ModelSection;
