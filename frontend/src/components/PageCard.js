import { Box, Container } from "@chakra-ui/react";
export default function PageCard({ children }) {
  return (
    <Container maxW="640px" py={10}>
      <Box bg="white" borderRadius="xl" boxShadow="sm" p={6}>
        {children}
      </Box>
    </Container>
  );
}