import React from "react";
import { Typography, Paper } from "@mui/material";
import { CheckCircle } from "@mui/icons-material";
import { useMediaQuery, useTheme } from "@mui/material";

const RealImageMessage = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("md"));

  return (
    <Paper
      sx={{
        p: isMobile ? 2 : 4,
        bgcolor: "success.dark",
        borderRadius: 2,
        textAlign: "center",
        border: "1px solid",
        borderColor: "success.main",
        mt: 3,
        minHeight: isMobile ? 150 : 200,
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <CheckCircle
        sx={{ fontSize: isMobile ? 50 : 60, color: "success.main", mb: 2 }}
      />
      <Typography
        variant={isMobile ? "h6" : "h5"}
        color="success.main"
        gutterBottom
        fontWeight="bold"
      >
        Authentic Image Detected
      </Typography>
      <Typography
        variant={isMobile ? "body2" : "body1"}
        color="text.secondary"
        sx={{
          fontSize: isMobile ? "1rem" : "1.1rem",
          lineHeight: 1.6,
        }}
      >
        Our AI model has determined this image appears to be real and
        unmodified.
      </Typography>
    </Paper>
  );
};

export default RealImageMessage;
