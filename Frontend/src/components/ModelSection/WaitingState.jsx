import React from "react";
import { Box, Typography, useMediaQuery, useTheme } from "@mui/material";
import { Analytics } from "@mui/icons-material";

const WaitingState = () => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down("md"));

  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        textAlign: "center",
        width: isMobile ? "100%" : "25rem",
        height: isMobile ? "20rem" : "29.5rem", // Smaller height on mobile
        flexShrink: 0,
      }}
    >
      <Box>
        <Analytics
          sx={{
            fontSize: isMobile ? 60 : 80,
            color: "text.disabled",
            mb: 2,
          }}
        />
        <Typography
          variant={isMobile ? "body1" : "h6"}
          color="text.secondary"
          gutterBottom
          fontWeight="medium"
        >
          Ready for Analysis
        </Typography>
        <Typography
          variant="body2"
          color="text.secondary"
          sx={{
            maxWidth: isMobile ? 280 : 400,
            mx: "auto",
            lineHeight: 1.5,
          }}
        >
          Upload an image to see detailed detection results
        </Typography>
      </Box>
    </Box>
  );
};

export default WaitingState;
