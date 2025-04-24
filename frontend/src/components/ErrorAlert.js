import { Alert, AlertIcon, AlertTitle, AlertDescription } from "@chakra-ui/react";
export default function ErrorAlert({ message }) {
  return (
    <Alert status="error" mt={4}>
      <AlertIcon />
      <AlertTitle mr={2}>Error</AlertTitle>
      <AlertDescription>{message}</AlertDescription>
    </Alert>
  );
}