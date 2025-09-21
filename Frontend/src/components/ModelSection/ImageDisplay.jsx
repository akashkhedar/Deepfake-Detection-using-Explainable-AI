import { Box, Button, useMediaQuery, useTheme } from "@mui/material";
import PropTypes from "prop-types";

const ImageDisplay = ({ image, isAnalyzing, onReset }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("md"));

  return (
    <Box sx={{ height: "100%", display: "flex", flexDirection: "column" }}>
      <Box
        sx={{
          position: "relative",
          width: isMobile ? "100%" : "25rem",
          height: isMobile ? "20rem" : "25rem", // Smaller height on mobile
          borderRadius: 3,
          overflow: "hidden",
          border: "2px solid",
          borderColor: "primary.main",
          mb: 2,
          bgcolor: "black",
          flexShrink: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <Box
          component="img"
          src={image}
          alt="Selected for analysis"
          sx={{
            width: "100%",
            height: "100%",
            objectFit: "contain",
            display: "block",
          }}
        />
      </Box>

      <Button
        variant="outlined"
        fullWidth
        onClick={onReset}
        sx={{ mt: "auto", flexShrink: 0 }}
        disabled={isAnalyzing}
      >
        Upload Different Image
      </Button>
    </Box>
  );
};

ImageDisplay.propTypes = {
  image: PropTypes.string.isRequired,
  prediction: PropTypes.oneOf(["real", "fake", null]),
  isAnalyzing: PropTypes.bool.isRequired,
  onReset: PropTypes.func.isRequired,
};

export default ImageDisplay;
