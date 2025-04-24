import { extendTheme } from "@chakra-ui/react";

const theme = extendTheme({
  fonts: {
    heading: `'Inter', sans-serif`,
    body: `'Inter', sans-serif`,
  },
  styles: {
    global: {
      "html, body": { bg: "gray.50", color: "gray.800" },
    },
  },
  components: {
    Button: { baseStyle: { borderRadius: "xl", fontWeight: 500 } },
    Input:  { baseStyle: { borderRadius: "xl" } },
  },
});

export default theme;