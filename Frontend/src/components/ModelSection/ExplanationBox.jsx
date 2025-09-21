import React from "react";
import { Box, Typography, Paper } from "@mui/material";
import { Psychology, Error } from "@mui/icons-material";
import { useMediaQuery, useTheme } from "@mui/material";
import PropTypes from "prop-types";

const ExplanationBox = ({ explanation, isAnalyzing, prediction }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("md"));
  const isError = prediction === "error";

  return (
    <Paper
      sx={{
        p: isMobile ? 2 : 4,
        bgcolor: isError ? "warning.dark" : "error.dark",
        borderRadius: 2,
        border: "1px solid",
        borderColor: isError ? "warning.main" : "error.main",
        mt: 3,
        minHeight: isMobile ? 200 : 250,
      }}
    >
      <Box sx={{ display: "flex", alignItems: "center", mb: isMobile ? 2 : 3 }}>
        {isError ? (
          <Error
            sx={{ fontSize: isMobile ? 28 : 32, color: "warning.main", mr: 2 }}
          />
        ) : (
          <Psychology
            sx={{ fontSize: isMobile ? 28 : 32, color: "error.main", mr: 2 }}
          />
        )}
        <Typography
          variant={isMobile ? "h6" : "h5"}
          color={isError ? "warning.main" : "error.main"}
        >
          {isError ? "Analysis Error" : "Why is this image fake?"}
        </Typography>
      </Box>
      {explanation ? (
        <Typography
          variant={isMobile ? "body2" : "body1"}
          sx={{
            lineHeight: 1.8,
            fontSize: isMobile ? "1rem" : "1.1rem",
            wordBreak: "break-word",
            overflowWrap: "break-word",
          }}
        >
          {explanation}
        </Typography>
      ) : isAnalyzing ? (
        <Typography
          variant="body2"
          color="text.secondary"
          sx={{ fontStyle: "italic" }}
        >
          {isError
            ? "Analyzing image..."
            : "Generating detailed explanation..."}
        </Typography>
      ) : null}
    </Paper>
  );
};

ExplanationBox.propTypes = {
  explanation: PropTypes.string,
  isAnalyzing: PropTypes.bool.isRequired,
  prediction: PropTypes.string,
};

export default ExplanationBox;
